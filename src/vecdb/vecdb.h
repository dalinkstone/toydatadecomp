/*
 * vecdb.h — Purpose-built vector database for two-tower model embeddings
 *
 * Stores 10M customer + 10K product embedding vectors (float32) and supports
 * fast top-K cosine similarity queries using Apple Accelerate BLAS.
 *
 * All vectors are L2-normalized on insert so dot product = cosine similarity.
 * Queries are read-only and safe to call from multiple threads concurrently.
 * Inserts are mutex-protected.
 */

#ifndef VECDB_H
#define VECDB_H

#include <stdint.h>

typedef struct VecDB VecDB;

/* Create a new vector database.
 * capacity: max vectors before realloc, dims: embedding dimensionality. */
VecDB *vecdb_create(uint32_t capacity, uint16_t dims);

/* Free the database and all associated memory. */
void vecdb_destroy(VecDB *db);

/* Insert a single vector. vec must point to `dims` floats.
 * The vector is L2-normalized internally. Returns 0 on success. */
int vecdb_insert(VecDB *db, uint32_t id, const float *vec);

/* Batch insert n vectors from contiguous arrays.
 * vecs: n × dims floats (row-major), ids: n uint32_t IDs. */
int vecdb_batch_insert(VecDB *db, const uint32_t *ids,
                       const float *vecs, uint32_t n);

/* Find top-k most similar vectors to query (cosine similarity).
 * Results written to out_ids[k] and out_scores[k], sorted descending. */
int vecdb_query_topk(const VecDB *db, const float *query, uint32_t k,
                     uint32_t *out_ids, float *out_scores);

/* Batch query: for each of n query vectors, find top-k similar.
 * out_ids[n×k] and out_scores[n×k], each query's results sorted descending.
 * Uses cblas_sgemm with chunked processing + GCD-parallel heap extraction. */
int vecdb_batch_query_topk(const VecDB *db, const float *queries, uint32_t n,
                           uint32_t k, uint32_t *out_ids, float *out_scores);

/* Save database to disk.
 * Binary format: [magic "VCDB" 4B][version u32][count u32][dims u16]
 *                [reserved 6B][ids: count×u32][vecs: count×dims×f32] */
int vecdb_save(const VecDB *db, const char *path);

/* Load database from disk. Returns NULL on failure. */
VecDB *vecdb_load(const char *path);

/* Accessors */
uint32_t vecdb_count(const VecDB *db);
uint16_t vecdb_dims(const VecDB *db);

#endif /* VECDB_H */
