/*
 * test_vecdb.c — Correctness tests + performance benchmarks for vecdb
 *
 * Tests:  batch insert, single query sorted, self-query ≈ 1.0,
 *         save/load round-trip, batch query sorted.
 * Bench:  insert throughput, 1000 single queries, 10K batch query.
 *
 * Compile: clang -O3 -march=native -DACCELERATE_NEW_LAPACK \
 *          -o test_vecdb test_vecdb.c vecdb.o -framework Accelerate -lm
 * Run:     ./test_vecdb
 */

#include "vecdb.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <sys/stat.h>

/* ------------------------------------------------------------------ */
/*  Config                                                             */
/* ------------------------------------------------------------------ */

#define NUM_VECS     100000
#define DIMS         256
#define TOP_K        10
#define SINGLE_RUNS  1000
#define BATCH_N      10000

/* ------------------------------------------------------------------ */
/*  Fast xorshift64 PRNG                                               */
/* ------------------------------------------------------------------ */

static uint64_t rng_state;

static void rng_seed(uint64_t s) { rng_state = s ? s : 1; }

static uint64_t xorshift64(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

static float randf(void) {
    return (float)(xorshift64() & 0xFFFFFF) / (float)0xFFFFFF - 0.5f;
}

/* ------------------------------------------------------------------ */
/*  Timer                                                              */
/* ------------------------------------------------------------------ */

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ------------------------------------------------------------------ */
/*  Main                                                               */
/* ------------------------------------------------------------------ */

int main(void) {
    printf("=== VecDB Test & Benchmark ===\n");
    printf("Vectors: %d   Dims: %d   Top-K: %d\n\n", NUM_VECS, DIMS, TOP_K);

    /* ---- Generate random data ---- */
    rng_seed(42);
    float    *vecs = malloc((uint64_t)NUM_VECS * DIMS * sizeof(float));
    uint32_t *ids  = malloc((uint64_t)NUM_VECS * sizeof(uint32_t));
    assert(vecs && ids);
    for (uint32_t i = 0; i < NUM_VECS; i++) {
        ids[i] = i + 1;   /* 1-based IDs */
        for (int d = 0; d < DIMS; d++)
            vecs[(uint64_t)i * DIMS + d] = randf();
    }

    /* ================================================================ */
    /*  1. Batch insert                                                  */
    /* ================================================================ */
    VecDB *db = vecdb_create(NUM_VECS, DIMS);
    assert(db);

    double t0 = now();
    assert(vecdb_batch_insert(db, ids, vecs, NUM_VECS) == 0);
    double insert_sec = now() - t0;

    assert(vecdb_count(db) == NUM_VECS);
    assert(vecdb_dims(db)  == DIMS);
    printf("[insert]   %d vectors in %.2f ms  (%.0f vec/s)\n",
           NUM_VECS, insert_sec * 1e3, NUM_VECS / insert_sec);

    /* ================================================================ */
    /*  2. Single query — correctness                                    */
    /* ================================================================ */

    /* 2a. Self-query: query with vector[0], expect it as top-1, score ≈ 1.0 */
    uint32_t out_ids[TOP_K];
    float    out_scores[TOP_K];

    assert(vecdb_query_topk(db, vecs, TOP_K, out_ids, out_scores) == 0);
    assert(out_ids[0] == 1);  /* ID of first inserted vector */
    assert(out_scores[0] > 0.999f);
    printf("[self-qry] top-1 id=%u  score=%.6f  (expected ~1.0)\n",
           out_ids[0], out_scores[0]);

    /* 2b. Random query — verify descending sort */
    rng_seed(999);
    float query[DIMS];
    for (int d = 0; d < DIMS; d++) query[d] = randf();

    assert(vecdb_query_topk(db, query, TOP_K, out_ids, out_scores) == 0);
    printf("[query]    top-%d results:\n", TOP_K);
    for (int i = 0; i < TOP_K; i++) {
        printf("           #%2d  id=%-8u  score=%.6f\n",
               i + 1, out_ids[i], out_scores[i]);
        if (i > 0) assert(out_scores[i - 1] >= out_scores[i]);
    }
    printf("           sorted descending: OK\n");

    /* ================================================================ */
    /*  3. Single query — benchmark (1000 queries)                       */
    /* ================================================================ */
    uint32_t bench_ids[TOP_K];
    float    bench_sc[TOP_K];
    t0 = now();
    for (int r = 0; r < SINGLE_RUNS; r++) {
        for (int d = 0; d < DIMS; d++) query[d] = randf();
        vecdb_query_topk(db, query, TOP_K, bench_ids, bench_sc);
    }
    double single_sec = now() - t0;
    double avg_ms = single_sec * 1e3 / SINGLE_RUNS;
    printf("[bench]    %d single queries in %.1f ms  (avg %.3f ms/query)\n",
           SINGLE_RUNS, single_sec * 1e3, avg_ms);

    /* ================================================================ */
    /*  4. Batch query — 10K queries                                     */
    /* ================================================================ */
    float    *bq     = malloc((uint64_t)BATCH_N * DIMS * sizeof(float));
    uint32_t *b_ids  = malloc((uint64_t)BATCH_N * TOP_K * sizeof(uint32_t));
    float    *b_sc   = malloc((uint64_t)BATCH_N * TOP_K * sizeof(float));
    assert(bq && b_ids && b_sc);

    rng_seed(12345);
    for (uint32_t i = 0; i < BATCH_N; i++)
        for (int d = 0; d < DIMS; d++)
            bq[(uint64_t)i * DIMS + d] = randf();

    t0 = now();
    assert(vecdb_batch_query_topk(db, bq, BATCH_N, TOP_K, b_ids, b_sc) == 0);
    double batch_sec = now() - t0;

    /* Verify all batch results sorted descending */
    for (uint32_t q = 0; q < BATCH_N; q++)
        for (int i = 1; i < TOP_K; i++)
            assert(b_sc[(uint64_t)q * TOP_K + i - 1] >=
                   b_sc[(uint64_t)q * TOP_K + i]);

    printf("[batch]    %d queries in %.1f ms  (%.0f queries/s)\n",
           BATCH_N, batch_sec * 1e3, BATCH_N / batch_sec);
    printf("           all results sorted: OK\n");

    /* ================================================================ */
    /*  5. Save / Load / Verify                                          */
    /* ================================================================ */
    const char *path = "/tmp/vecdb_test.bin";
    assert(vecdb_save(db, path) == 0);

    struct stat st;
    stat(path, &st);
    printf("[save]     %s  (%.1f MB)\n", path, st.st_size / (1024.0 * 1024.0));

    VecDB *db2 = vecdb_load(path);
    assert(db2);
    assert(vecdb_count(db2) == vecdb_count(db));
    assert(vecdb_dims(db2)  == vecdb_dims(db));

    /* Same query on loaded DB must produce equivalent results.
     * BLAS may use different internal accumulation order between runs
     * (thread scheduling), so we compare scores with tolerance. */
    rng_seed(999);
    for (int d = 0; d < DIMS; d++) query[d] = randf();

    uint32_t chk_ids[TOP_K];
    float    chk_sc[TOP_K];
    assert(vecdb_query_topk(db2, query, TOP_K, chk_ids, chk_sc) == 0);
    for (int i = 0; i < TOP_K; i++) {
        assert(chk_ids[i] == out_ids[i]);
        assert(fabsf(chk_sc[i] - out_scores[i]) < 1e-5f);
    }
    /* Verify loaded results are also sorted */
    for (int i = 1; i < TOP_K; i++)
        assert(chk_sc[i - 1] >= chk_sc[i]);
    printf("[load]     query results match original: OK\n");

    /* ================================================================ */
    /*  Cleanup                                                          */
    /* ================================================================ */
    vecdb_destroy(db);
    vecdb_destroy(db2);
    free(vecs); free(ids);
    free(bq); free(b_ids); free(b_sc);
    remove(path);

    /* ================================================================ */
    /*  Summary                                                          */
    /* ================================================================ */
    printf("\n=== Summary ===\n");
    printf("Insert:  %.0f vectors/sec\n", NUM_VECS / insert_sec);
    printf("Query:   %.3f ms avg  (%d single queries)\n", avg_ms, SINGLE_RUNS);
    printf("Batch:   %.0f queries/sec  (%d queries, top-%d)\n",
           BATCH_N / batch_sec, BATCH_N, TOP_K);
    printf("\nAll tests passed.\n");
    return 0;
}
