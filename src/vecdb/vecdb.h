/*
 * vecdb.h — Lightweight vector database for item embeddings
 *
 * Stores dense float vectors and supports fast approximate nearest neighbor
 * search using brute-force dot product with Apple Accelerate SIMD.
 *
 * Used at inference time to find top-K product recommendations for each user
 * by comparing user embeddings against pre-computed item embeddings.
 */

#ifndef VECDB_H
#define VECDB_H

#include <stddef.h>
#include <stdint.h>

/* Vector database handle */
typedef struct {
    float *data;       /* Contiguous array of all vectors: [num_vectors × dim] */
    int32_t *ids;      /* Mapping from index to external ID */
    size_t num_vectors;
    size_t dim;
    size_t capacity;
} vecdb_t;

/* Top-K result entry */
typedef struct {
    int32_t id;
    float score;
} vecdb_result_t;

/* Initialize a vector database with given dimension and initial capacity */
int vecdb_init(vecdb_t *db, size_t dim, size_t initial_capacity);

/* Free all resources */
void vecdb_free(vecdb_t *db);

/* Add a vector with an external ID. Returns 0 on success. */
int vecdb_add(vecdb_t *db, int32_t id, const float *vector);

/* Find top-K nearest vectors by dot product similarity.
 * Results are written to `results` (must have space for `k` entries).
 * Returns the number of results written (<= k). */
size_t vecdb_topk(const vecdb_t *db, const float *query, size_t k,
                  vecdb_result_t *results);

/* Batch top-K: query multiple vectors at once (for throughput).
 * `queries` is [num_queries × dim], results is [num_queries × k]. */
size_t vecdb_topk_batch(const vecdb_t *db, const float *queries,
                        size_t num_queries, size_t k,
                        vecdb_result_t *results);

/* Save/load database to/from file */
int vecdb_save(const vecdb_t *db, const char *path);
int vecdb_load(vecdb_t *db, const char *path);

#endif /* VECDB_H */
