/*
 * vecdb.c — Vector database implementation using Apple Accelerate
 *
 * Uses vDSP_dotpr for SIMD-accelerated dot products on Apple Silicon.
 * Brute-force search with partial sort for top-K retrieval.
 *
 * Compile: clang -O3 -march=native -c -o vecdb.o vecdb.c -framework Accelerate
 */

#include "vecdb.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <Accelerate/Accelerate.h>

int vecdb_init(vecdb_t *db, size_t dim, size_t initial_capacity) {
    db->dim = dim;
    db->capacity = initial_capacity;
    db->num_vectors = 0;
    db->data = (float *)malloc(initial_capacity * dim * sizeof(float));
    db->ids = (int32_t *)malloc(initial_capacity * sizeof(int32_t));
    if (!db->data || !db->ids) {
        vecdb_free(db);
        return -1;
    }
    return 0;
}

void vecdb_free(vecdb_t *db) {
    free(db->data);
    free(db->ids);
    db->data = NULL;
    db->ids = NULL;
    db->num_vectors = 0;
    db->capacity = 0;
}

int vecdb_add(vecdb_t *db, int32_t id, const float *vector) {
    if (db->num_vectors >= db->capacity) {
        size_t new_cap = db->capacity * 2;
        float *new_data = (float *)realloc(db->data, new_cap * db->dim * sizeof(float));
        int32_t *new_ids = (int32_t *)realloc(db->ids, new_cap * sizeof(int32_t));
        if (!new_data || !new_ids) return -1;
        db->data = new_data;
        db->ids = new_ids;
        db->capacity = new_cap;
    }
    memcpy(db->data + db->num_vectors * db->dim, vector, db->dim * sizeof(float));
    db->ids[db->num_vectors] = id;
    db->num_vectors++;
    return 0;
}

/* Min-heap for top-K tracking */
static void heap_sift_down(vecdb_result_t *heap, size_t n, size_t i) {
    while (1) {
        size_t smallest = i;
        size_t l = 2 * i + 1;
        size_t r = 2 * i + 2;
        if (l < n && heap[l].score < heap[smallest].score) smallest = l;
        if (r < n && heap[r].score < heap[smallest].score) smallest = r;
        if (smallest == i) break;
        vecdb_result_t tmp = heap[i];
        heap[i] = heap[smallest];
        heap[smallest] = tmp;
        i = smallest;
    }
}

size_t vecdb_topk(const vecdb_t *db, const float *query, size_t k,
                  vecdb_result_t *results) {
    if (k > db->num_vectors) k = db->num_vectors;

    /* Initialize min-heap with first k results */
    for (size_t i = 0; i < k; i++) {
        float score;
        vDSP_dotpr(query, 1, db->data + i * db->dim, 1, &score, (vDSP_Length)db->dim);
        results[i].id = db->ids[i];
        results[i].score = score;
    }
    /* Build heap */
    for (int i = (int)k / 2 - 1; i >= 0; i--) {
        heap_sift_down(results, k, (size_t)i);
    }

    /* Scan remaining vectors */
    for (size_t i = k; i < db->num_vectors; i++) {
        float score;
        vDSP_dotpr(query, 1, db->data + i * db->dim, 1, &score, (vDSP_Length)db->dim);
        if (score > results[0].score) {
            results[0].id = db->ids[i];
            results[0].score = score;
            heap_sift_down(results, k, 0);
        }
    }

    return k;
}

size_t vecdb_topk_batch(const vecdb_t *db, const float *queries,
                        size_t num_queries, size_t k,
                        vecdb_result_t *results) {
    for (size_t q = 0; q < num_queries; q++) {
        vecdb_topk(db, queries + q * db->dim, k, results + q * k);
    }
    return num_queries * k;
}

int vecdb_save(const vecdb_t *db, const char *path) {
    FILE *fp = fopen(path, "wb");
    if (!fp) return -1;
    fwrite(&db->num_vectors, sizeof(size_t), 1, fp);
    fwrite(&db->dim, sizeof(size_t), 1, fp);
    fwrite(db->ids, sizeof(int32_t), db->num_vectors, fp);
    fwrite(db->data, sizeof(float), db->num_vectors * db->dim, fp);
    fclose(fp);
    return 0;
}

int vecdb_load(vecdb_t *db, const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return -1;
    size_t num, dim;
    fread(&num, sizeof(size_t), 1, fp);
    fread(&dim, sizeof(size_t), 1, fp);
    vecdb_init(db, dim, num);
    db->num_vectors = num;
    fread(db->ids, sizeof(int32_t), num, fp);
    fread(db->data, sizeof(float), num * dim, fp);
    fclose(fp);
    return 0;
}
