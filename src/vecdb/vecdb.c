/*
 * vecdb.c — Purpose-built vector database for two-tower model embeddings
 *
 * Optimized for Apple Silicon (M4 Max):
 *   - cblas_sgemv  for single-query dot products  (matrix × vector)
 *   - cblas_sgemm  for batch-query dot products    (matrix × matrix)
 *   - dispatch_apply_f for parallel heap extraction (GCD)
 *   - posix_memalign for 64-byte aligned vector storage
 *   - L2 normalization on insert → dot product = cosine similarity
 *
 * Compile: clang -O3 -march=native -c -o vecdb.o vecdb.c -framework Accelerate
 */

#include "vecdb.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <pthread.h>

#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>

/* ------------------------------------------------------------------ */
/*  Constants                                                          */
/* ------------------------------------------------------------------ */

#define VECDB_ALIGN         64                      /* cache-line align  */
#define MAX_SCORE_BUF_BYTES (256ULL * 1024 * 1024)  /* 256 MB per chunk  */
#define VECDB_VERSION       1
#define SMALL_BATCH_THRESH  4   /* below this, loop single queries */

/* ------------------------------------------------------------------ */
/*  Internal structure                                                 */
/* ------------------------------------------------------------------ */

struct VecDB {
    float          *vecs;       /* aligned: capacity × dims floats       */
    uint32_t       *ids;        /* capacity uint32_t IDs                 */
    uint32_t        count;      /* stored vectors                        */
    uint32_t        capacity;   /* allocated slots                       */
    uint16_t        dims;       /* embedding dimensionality              */
    pthread_mutex_t lock;       /* protects inserts; queries are lockfree*/
};

/* ------------------------------------------------------------------ */
/*  File header — 20 bytes, packed                                     */
/* ------------------------------------------------------------------ */

#pragma pack(push, 1)
typedef struct {
    uint8_t  magic[4];     /* "VCDB"          */
    uint32_t version;      /* 1               */
    uint32_t count;
    uint16_t dims;
    uint8_t  reserved[6];  /* zero-filled     */
} FileHeader;
#pragma pack(pop)

_Static_assert(sizeof(FileHeader) == 20, "FileHeader must be exactly 20 bytes");

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

static void *aligned_malloc(size_t align, size_t bytes) {
    if (bytes == 0) return NULL;
    void *p = NULL;
    if (posix_memalign(&p, align, bytes) != 0) return NULL;
    return p;
}

/* L2-normalize in place.  Zero-length vectors are left unchanged. */
static inline void normalize(float *v, int dims) {
    float norm = cblas_snrm2(dims, v, 1);
    if (norm > 1e-10f) {
        float s = 1.0f / norm;
        cblas_sscal(dims, s, v, 1);
    }
}

/* ------------------------------------------------------------------ */
/*  Min-heap for top-K selection                                       */
/* ------------------------------------------------------------------ */

typedef struct { uint32_t id; float score; } HE;  /* heap entry */

static inline void heap_sift(HE *h, uint32_t n, uint32_t i) {
    for (;;) {
        uint32_t s = i, l = 2*i + 1, r = l + 1;
        if (l < n && h[l].score < h[s].score) s = l;
        if (r < n && h[r].score < h[s].score) s = r;
        if (s == i) return;
        HE t = h[i]; h[i] = h[s]; h[s] = t;
        i = s;
    }
}

/* Build min-heap from first k elements. */
static void heap_build(HE *h, uint32_t k,
                       const float *scores, const uint32_t *ids) {
    for (uint32_t i = 0; i < k; i++) {
        h[i].id    = ids[i];
        h[i].score = scores[i];
    }
    if (k < 2) return;
    for (int32_t i = (int32_t)(k / 2) - 1; i >= 0; i--)
        heap_sift(h, k, (uint32_t)i);
}

/* Scan scores and push any that beat the current minimum. */
static void heap_scan(HE *h, uint32_t k,
                      const float *scores, const uint32_t *ids, uint32_t n) {
    float min_score = h[0].score;
    for (uint32_t i = 0; i < n; i++) {
        if (scores[i] > min_score) {
            h[0].id    = ids[i];
            h[0].score = scores[i];
            heap_sift(h, k, 0);
            min_score = h[0].score;
        }
    }
}

/* Extract heap into descending-sorted output arrays.  Destroys heap. */
static void heap_extract_desc(HE *h, uint32_t k,
                              uint32_t *out_ids, float *out_scores) {
    for (uint32_t sz = k; sz > 0; sz--) {
        out_ids[sz - 1]    = h[0].id;
        out_scores[sz - 1] = h[0].score;
        h[0] = h[sz - 1];
        if (sz > 1) heap_sift(h, sz - 1, 0);
    }
}

/* ------------------------------------------------------------------ */
/*  Create / Destroy                                                   */
/* ------------------------------------------------------------------ */

VecDB *vecdb_create(uint32_t capacity, uint16_t dims) {
    if (dims == 0) return NULL;

    VecDB *db = calloc(1, sizeof *db);
    if (!db) return NULL;

    db->dims     = dims;
    db->capacity = capacity > 0 ? capacity : 1024;

    uint64_t vb = (uint64_t)db->capacity * dims * sizeof(float);
    uint64_t ib = (uint64_t)db->capacity * sizeof(uint32_t);

    db->vecs = aligned_malloc(VECDB_ALIGN, vb);
    db->ids  = malloc(ib);
    if (!db->vecs || !db->ids) {
        free(db->vecs); free(db->ids); free(db);
        return NULL;
    }
    pthread_mutex_init(&db->lock, NULL);
    return db;
}

void vecdb_destroy(VecDB *db) {
    if (!db) return;
    pthread_mutex_destroy(&db->lock);
    free(db->vecs);
    free(db->ids);
    free(db);
}

/* ------------------------------------------------------------------ */
/*  Capacity management (caller must hold lock)                        */
/* ------------------------------------------------------------------ */

static int grow(VecDB *db, uint32_t need) {
    uint64_t req = (uint64_t)db->count + need;
    if (req <= db->capacity) return 0;

    uint32_t nc = db->capacity;
    while ((uint64_t)nc < req) nc = nc < 1024 ? 1024 : nc * 2;

    float    *nv = aligned_malloc(VECDB_ALIGN, (uint64_t)nc * db->dims * sizeof(float));
    uint32_t *ni = malloc((uint64_t)nc * sizeof(uint32_t));
    if (!nv || !ni) { free(nv); free(ni); return -1; }

    if (db->count > 0) {
        memcpy(nv, db->vecs, (uint64_t)db->count * db->dims * sizeof(float));
        memcpy(ni, db->ids,  (uint64_t)db->count * sizeof(uint32_t));
    }
    free(db->vecs); free(db->ids);
    db->vecs     = nv;
    db->ids      = ni;
    db->capacity = nc;
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Insert                                                             */
/* ------------------------------------------------------------------ */

int vecdb_insert(VecDB *db, uint32_t id, const float *vec) {
    if (!db || !vec) return -1;
    pthread_mutex_lock(&db->lock);

    if (grow(db, 1) != 0) { pthread_mutex_unlock(&db->lock); return -1; }

    float *dst = db->vecs + (uint64_t)db->count * db->dims;
    memcpy(dst, vec, (uint64_t)db->dims * sizeof(float));
    normalize(dst, (int)db->dims);
    db->ids[db->count] = id;
    db->count++;

    pthread_mutex_unlock(&db->lock);
    return 0;
}

int vecdb_batch_insert(VecDB *db, const uint32_t *ids,
                       const float *vecs, uint32_t n) {
    if (!db || !ids || !vecs || n == 0) return -1;
    pthread_mutex_lock(&db->lock);

    if (grow(db, n) != 0) { pthread_mutex_unlock(&db->lock); return -1; }

    uint64_t off = (uint64_t)db->count * db->dims;
    memcpy(db->vecs + off, vecs, (uint64_t)n * db->dims * sizeof(float));
    memcpy(db->ids + db->count, ids, (uint64_t)n * sizeof(uint32_t));

    /* Normalize each new vector in place */
    int d = (int)db->dims;
    for (uint32_t i = 0; i < n; i++)
        normalize(db->vecs + off + (uint64_t)i * db->dims, d);

    db->count += n;
    pthread_mutex_unlock(&db->lock);
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Single query  (sgemv + min-heap)                                   */
/* ------------------------------------------------------------------ */

int vecdb_query_topk(const VecDB *db, const float *query, uint32_t k,
                     uint32_t *out_ids, float *out_scores) {
    if (!db || !query || !out_ids || !out_scores) return -1;
    if (k == 0) return 0;
    if (db->count == 0) {
        memset(out_ids,    0, (size_t)k * sizeof(uint32_t));
        memset(out_scores, 0, (size_t)k * sizeof(float));
        return 0;
    }

    uint32_t ak = k < db->count ? k : db->count;
    int d = (int)db->dims;

    /* Normalize a copy of the query */
    float *nq = aligned_malloc(VECDB_ALIGN, (uint64_t)db->dims * sizeof(float));
    if (!nq) return -1;
    memcpy(nq, query, (size_t)db->dims * sizeof(float));
    normalize(nq, d);

    /* scores = vecs × nq   (N×D matrix  ×  D×1 vector → N×1) */
    float *scores = aligned_malloc(VECDB_ALIGN, (uint64_t)db->count * sizeof(float));
    if (!scores) { free(nq); return -1; }

    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                (int)db->count, d,
                1.0f, db->vecs, d,
                nq, 1,
                0.0f, scores, 1);

    /* Top-K via min-heap */
    HE *heap = malloc((size_t)ak * sizeof(HE));
    if (!heap) { free(scores); free(nq); return -1; }

    heap_build(heap, ak, scores, db->ids);
    if (db->count > ak)
        heap_scan(heap, ak, scores + ak, db->ids + ak, db->count - ak);
    heap_extract_desc(heap, ak, out_ids, out_scores);

    for (uint32_t i = ak; i < k; i++) { out_ids[i] = 0; out_scores[i] = 0.0f; }

    free(heap); free(scores); free(nq);
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Batch query  (chunked sgemm + GCD-parallel heap processing)        */
/* ------------------------------------------------------------------ */

/* Context shared across GCD work items within one chunk iteration. */
typedef struct {
    HE             *heaps;     /* n × ak heap entries              */
    const float    *scores;    /* n × stride score buffer          */
    const uint32_t *db_ids;    /* db->ids base pointer             */
    uint32_t        ak;        /* actual k                         */
    uint32_t        stride;    /* allocated leading dim (= chunk)  */
    uint32_t        batch;     /* vectors in this chunk            */
    uint32_t        offset;    /* start index into db              */
    int             first;     /* 1 → initialise heaps             */
} ChunkCtx;

static void chunk_work(void *ctx, size_t q) {
    ChunkCtx *c  = ctx;
    HE *h        = c->heaps + q * c->ak;
    const float *row = c->scores + q * c->stride;

    if (c->first) {
        heap_build(h, c->ak, row, c->db_ids);
        if (c->batch > c->ak)
            heap_scan(h, c->ak, row + c->ak, c->db_ids + c->ak,
                      c->batch - c->ak);
    } else {
        heap_scan(h, c->ak, row, c->db_ids + c->offset, c->batch);
    }
}

int vecdb_batch_query_topk(const VecDB *db, const float *queries, uint32_t n,
                           uint32_t k, uint32_t *out_ids, float *out_scores) {
    if (!db || !queries || !out_ids || !out_scores) return -1;
    if (n == 0 || k == 0) return 0;
    if (db->count == 0) {
        memset(out_ids,    0, (uint64_t)n * k * sizeof(uint32_t));
        memset(out_scores, 0, (uint64_t)n * k * sizeof(float));
        return 0;
    }

    /* For tiny batches, just loop single queries (avoids sgemm overhead). */
    if (n <= SMALL_BATCH_THRESH) {
        for (uint32_t q = 0; q < n; q++) {
            int rc = vecdb_query_topk(db, queries + (uint64_t)q * db->dims, k,
                                      out_ids + (uint64_t)q * k,
                                      out_scores + (uint64_t)q * k);
            if (rc != 0) return rc;
        }
        return 0;
    }

    uint32_t ak = k < db->count ? k : db->count;
    int d = (int)db->dims;

    /* Normalize all queries into a private copy */
    uint64_t qbytes = (uint64_t)n * db->dims * sizeof(float);
    float *nq = aligned_malloc(VECDB_ALIGN, qbytes);
    if (!nq) return -1;
    memcpy(nq, queries, qbytes);
    for (uint32_t i = 0; i < n; i++)
        normalize(nq + (uint64_t)i * db->dims, d);

    /* Determine chunk size: score buffer ≤ MAX_SCORE_BUF_BYTES */
    uint32_t chunk = (uint32_t)(MAX_SCORE_BUF_BYTES / ((uint64_t)n * sizeof(float)));
    if (chunk > db->count) chunk = db->count;
    if (chunk < ak)        chunk = ak;

    /* Allocate working memory */
    float *scores = aligned_malloc(VECDB_ALIGN, (uint64_t)n * chunk * sizeof(float));
    HE    *heaps  = malloc((uint64_t)n * ak * sizeof(HE));
    if (!scores || !heaps) { free(nq); free(scores); free(heaps); return -1; }

    dispatch_queue_t gq = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);
    ChunkCtx ctx;
    ctx.heaps  = heaps;
    ctx.scores = scores;
    ctx.db_ids = db->ids;
    ctx.ak     = ak;
    ctx.stride = chunk;

    uint32_t offset = 0;
    int first = 1;

    while (offset < db->count) {
        uint32_t batch = db->count - offset;
        if (batch > chunk) batch = chunk;

        /*  scores[q][i] = dot(nq_q, vec_{offset+i})
         *  = nq (n×D)  ×  vecs_chunk^T (D×batch)  →  (n×batch)
         *  stored with leading dimension = chunk (stride). */
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    (int)n, (int)batch, d,
                    1.0f, nq, d,
                    db->vecs + (uint64_t)offset * db->dims, d,
                    0.0f, scores, (int)chunk);

        ctx.batch  = batch;
        ctx.offset = offset;
        ctx.first  = first;

        dispatch_apply_f((size_t)n, gq, &ctx, chunk_work);

        offset += batch;
        first   = 0;
    }

    /* Extract sorted results from heaps */
    for (uint32_t q = 0; q < n; q++) {
        heap_extract_desc(heaps + (uint64_t)q * ak, ak,
                          out_ids + (uint64_t)q * k,
                          out_scores + (uint64_t)q * k);
        for (uint32_t i = ak; i < k; i++) {
            out_ids[(uint64_t)q * k + i]    = 0;
            out_scores[(uint64_t)q * k + i] = 0.0f;
        }
    }

    free(heaps); free(scores); free(nq);
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Persistence                                                        */
/* ------------------------------------------------------------------ */

int vecdb_save(const VecDB *db, const char *path) {
    if (!db || !path) return -1;
    FILE *fp = fopen(path, "wb");
    if (!fp) return -1;

    FileHeader hdr;
    memset(&hdr, 0, sizeof hdr);
    hdr.magic[0] = 'V'; hdr.magic[1] = 'C';
    hdr.magic[2] = 'D'; hdr.magic[3] = 'B';
    hdr.version  = VECDB_VERSION;
    hdr.count    = db->count;
    hdr.dims     = db->dims;

    if (fwrite(&hdr, sizeof hdr, 1, fp) != 1) goto fail;
    if (db->count > 0) {
        if (fwrite(db->ids,  sizeof(uint32_t), db->count, fp) != db->count)
            goto fail;
        uint64_t nf = (uint64_t)db->count * db->dims;
        if (fwrite(db->vecs, sizeof(float), nf, fp) != nf)
            goto fail;
    }
    fclose(fp);
    return 0;
fail:
    fclose(fp);
    return -1;
}

VecDB *vecdb_load(const char *path) {
    if (!path) return NULL;
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;

    FileHeader hdr;
    if (fread(&hdr, sizeof hdr, 1, fp) != 1)                         goto fail;
    if (hdr.magic[0]!='V' || hdr.magic[1]!='C' ||
        hdr.magic[2]!='D' || hdr.magic[3]!='B')                      goto fail;
    if (hdr.version != VECDB_VERSION || hdr.dims == 0)                goto fail;

    VecDB *db = vecdb_create(hdr.count > 0 ? hdr.count : 1, hdr.dims);
    if (!db) goto fail;

    if (hdr.count > 0) {
        if (fread(db->ids, sizeof(uint32_t), hdr.count, fp) != hdr.count) {
            vecdb_destroy(db); goto fail;
        }
        uint64_t nf = (uint64_t)hdr.count * hdr.dims;
        if (fread(db->vecs, sizeof(float), nf, fp) != nf) {
            vecdb_destroy(db); goto fail;
        }
    }
    db->count = hdr.count;
    fclose(fp);
    return db;

fail:
    fclose(fp);
    return NULL;
}

/* ------------------------------------------------------------------ */
/*  Accessors                                                          */
/* ------------------------------------------------------------------ */

uint32_t vecdb_count(const VecDB *db) { return db ? db->count : 0; }
uint16_t vecdb_dims (const VecDB *db) { return db ? db->dims  : 0; }
