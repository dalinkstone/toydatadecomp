/*
 * txn_generator.c — High-performance synthetic transaction generator
 *
 * Generates 10B transactions (10M customers × 1,000 transactions each)
 * using multithreaded C with zstd-compressed output via popen().
 *
 * Each transaction record:
 *   txn_id, customer_id, store_id, product_id, quantity, unit_price,
 *   total_price, txn_date, txn_time, payment_method
 *
 * Output: data/synthetic/transactions/txns_NNNN.csv.zst
 *
 * Disk optimization: Raw CSV would be ~1.5TB. zstd compression via
 * popen("zstd -3 -o <file>", "w") reduces this to ~180GB while allowing
 * DuckDB to read .csv.zst files natively with predicate pushdown.
 *
 * Compile: clang -O3 -march=native -o txn_generator txn_generator.c -lm -lpthread
 * Run:     ./txn_generator
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#define NUM_CUSTOMERS    10000000
#define TXNS_PER_CUST    1000
#define NUM_STORES       9000
#define NUM_PRODUCTS     10000
#define NUM_THREADS      10
#define CUSTOMERS_PER_THREAD (NUM_CUSTOMERS / NUM_THREADS)
#define OUTPUT_DIR       "data/synthetic/transactions"

/* Thread-local PRNG (xoshiro256**) for fast, high-quality random numbers */
typedef struct {
    uint64_t s[4];
} rng_state_t;

static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static inline uint64_t rng_next(rng_state_t *rng) {
    const uint64_t result = rotl(rng->s[1] * 5, 7) * 9;
    const uint64_t t = rng->s[1] << 17;
    rng->s[2] ^= rng->s[0];
    rng->s[3] ^= rng->s[1];
    rng->s[1] ^= rng->s[2];
    rng->s[0] ^= rng->s[3];
    rng->s[2] ^= t;
    rng->s[3] = rotl(rng->s[3], 45);
    return result;
}

static void rng_seed(rng_state_t *rng, uint64_t seed) {
    /* SplitMix64 to initialize state */
    for (int i = 0; i < 4; i++) {
        seed += 0x9e3779b97f4a7c15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        rng->s[i] = z ^ (z >> 31);
    }
}

typedef struct {
    int thread_id;
    int customer_start;
    int customer_end;
} thread_arg_t;

static const char *payment_methods[] = {
    "cash", "credit", "debit", "extracare", "apple_pay"
};
#define NUM_PAYMENT_METHODS 5

void *generate_chunk(void *arg) {
    thread_arg_t *targ = (thread_arg_t *)arg;
    rng_state_t rng;
    rng_seed(&rng, (uint64_t)targ->thread_id * 1234567891ULL);

    char filename[256];
    snprintf(filename, sizeof(filename),
             "zstd -3 -o %s/txns_%04d.csv.zst", OUTPUT_DIR, targ->thread_id);

    FILE *fp = popen(filename, "w");
    if (!fp) {
        fprintf(stderr, "ERROR: Failed to open zstd pipe for thread %d\n", targ->thread_id);
        return NULL;
    }

    /* CSV header */
    fprintf(fp, "txn_id,customer_id,store_id,product_id,quantity,unit_price,"
                "total_price,txn_date,txn_time,payment_method\n");

    uint64_t txn_id_base = (uint64_t)targ->customer_start * TXNS_PER_CUST;

    for (int c = targ->customer_start; c < targ->customer_end; c++) {
        for (int t = 0; t < TXNS_PER_CUST; t++) {
            uint64_t txn_id = txn_id_base++;
            int store_id = (int)(rng_next(&rng) % NUM_STORES) + 1;
            int product_id = (int)(rng_next(&rng) % NUM_PRODUCTS) + 1;
            int quantity = (int)(rng_next(&rng) % 5) + 1;
            double unit_price = 0.99 + (double)(rng_next(&rng) % 5000) / 100.0;
            double total_price = unit_price * quantity;
            int day_offset = (int)(rng_next(&rng) % (365 * 3)); /* 3 years of data */
            int hour = (int)(rng_next(&rng) % 16) + 6; /* 6am - 10pm */
            int minute = (int)(rng_next(&rng) % 60);
            int second = (int)(rng_next(&rng) % 60);
            const char *payment = payment_methods[rng_next(&rng) % NUM_PAYMENT_METHODS];

            /* Date: offset from 2022-01-01 */
            int year = 2022 + day_offset / 365;
            int doy = day_offset % 365;
            int month = doy / 30 + 1;
            if (month > 12) month = 12;
            int day = doy % 30 + 1;

            fprintf(fp, "%llu,%d,%d,%d,%d,%.2f,%.2f,%04d-%02d-%02d,%02d:%02d:%02d,%s\n",
                    txn_id, c + 1, store_id, product_id, quantity,
                    unit_price, total_price, year, month, day,
                    hour, minute, second, payment);
        }
    }

    pclose(fp);
    printf("Thread %d complete: customers %d-%d\n",
           targ->thread_id, targ->customer_start + 1, targ->customer_end);
    return NULL;
}

int main(void) {
    printf("txn_generator: Generating %dB transactions (%dM customers × %d txns)\n",
           (int)((long long)NUM_CUSTOMERS * TXNS_PER_CUST / 1000000000LL),
           NUM_CUSTOMERS / 1000000, TXNS_PER_CUST);
    printf("Output: %s/txns_NNNN.csv.zst (zstd-compressed)\n", OUTPUT_DIR);
    printf("Threads: %d\n\n", NUM_THREADS);

    time_t start = time(NULL);

    pthread_t threads[NUM_THREADS];
    thread_arg_t args[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        args[i].thread_id = i;
        args[i].customer_start = i * CUSTOMERS_PER_THREAD;
        args[i].customer_end = (i + 1) * CUSTOMERS_PER_THREAD;
        pthread_create(&threads[i], NULL, generate_chunk, &args[i]);
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    time_t elapsed = time(NULL) - start;
    printf("\nDone in %ld seconds.\n", (long)elapsed);
    return 0;
}
