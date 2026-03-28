/*
 * test_vecdb.c — Unit tests for the vector database
 *
 * Tests basic operations: init, add, top-K search, save/load.
 * Verifies correctness of SIMD dot product via Accelerate framework.
 *
 * Compile: clang -O3 -march=native -o test_vecdb test_vecdb.c vecdb.o -framework Accelerate -lm
 * Run:     ./test_vecdb
 */

#include "vecdb.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define DIM 64
#define NUM_VECTORS 1000
#define TOP_K 10

static void fill_random(float *v, size_t n, unsigned int seed) {
    srand(seed);
    for (size_t i = 0; i < n; i++) {
        v[i] = (float)rand() / (float)RAND_MAX - 0.5f;
    }
}

static void test_init_and_add(void) {
    printf("test_init_and_add... ");
    vecdb_t db;
    assert(vecdb_init(&db, DIM, 16) == 0);

    float vec[DIM];
    for (int i = 0; i < NUM_VECTORS; i++) {
        fill_random(vec, DIM, (unsigned int)i);
        assert(vecdb_add(&db, i + 1, vec) == 0);
    }
    assert(db.num_vectors == NUM_VECTORS);

    vecdb_free(&db);
    printf("PASS\n");
}

static void test_topk(void) {
    printf("test_topk... ");
    vecdb_t db;
    vecdb_init(&db, DIM, NUM_VECTORS);

    float vec[DIM];
    for (int i = 0; i < NUM_VECTORS; i++) {
        fill_random(vec, DIM, (unsigned int)i);
        vecdb_add(&db, i + 1, vec);
    }

    /* Query with vector 0 — should find itself as top match */
    float query[DIM];
    fill_random(query, DIM, 0);

    vecdb_result_t results[TOP_K];
    size_t n = vecdb_topk(&db, query, TOP_K, results);
    assert(n == TOP_K);

    /* Results should be sorted by score (min-heap, so unsorted — just check top exists) */
    int found_self = 0;
    for (size_t i = 0; i < n; i++) {
        if (results[i].id == 1) found_self = 1;
    }
    assert(found_self);

    vecdb_free(&db);
    printf("PASS\n");
}

static void test_save_load(void) {
    printf("test_save_load... ");
    vecdb_t db;
    vecdb_init(&db, DIM, NUM_VECTORS);

    float vec[DIM];
    for (int i = 0; i < NUM_VECTORS; i++) {
        fill_random(vec, DIM, (unsigned int)i);
        vecdb_add(&db, i + 1, vec);
    }

    const char *path = "/tmp/test_vecdb.bin";
    assert(vecdb_save(&db, path) == 0);

    vecdb_t db2;
    assert(vecdb_load(&db2, path) == 0);
    assert(db2.num_vectors == db.num_vectors);
    assert(db2.dim == db.dim);

    /* Verify data integrity */
    for (size_t i = 0; i < db.num_vectors * db.dim; i++) {
        assert(fabsf(db.data[i] - db2.data[i]) < 1e-6f);
    }

    vecdb_free(&db);
    vecdb_free(&db2);
    remove(path);
    printf("PASS\n");
}

int main(void) {
    printf("=== vecdb unit tests ===\n");
    test_init_and_add();
    test_topk();
    test_save_load();
    printf("\nAll tests passed.\n");
    return 0;
}
