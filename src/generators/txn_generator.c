/*
 * txn_generator.c — High-performance multithreaded transaction generator
 *
 * Generates 10B transactions (10M customers × 1,000 txns each) with realistic
 * behavioral modeling: Zipf product popularity, seasonal patterns, shopping
 * trip clustering, time-of-day distributions, discounts, and state sales tax.
 *
 * Output: data/synthetic/transactions/txns_tNN.csv.zst (one per thread)
 *         Uses zstd compression via popen for ~10× disk savings.
 *
 * Compile: clang -O3 -march=native -o txn_generator src/generators/txn_generator.c -lm -lpthread
 * Run:     ./txn_generator [num_customers] [txns_per_customer] [num_products] [_] [output_dir] [stores_csv]
 * Default: ./txn_generator 10000000 1000 12000 0 data/synthetic/transactions data/real/stores.csv
 *
 * Store IDs are loaded from stores.csv (real CVS store numbers) so that
 * transaction.store_id correctly joins to the stores table.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <inttypes.h>
#include <sys/stat.h>

/* ========================================================================
 * Configuration
 * ======================================================================== */

static int    g_num_customers     = 10000000;
static int    g_txns_per_customer = 1000;
static int    g_num_products      = 12000;
static int    g_num_stores        = 9000;
static int    g_num_threads       = 8;
static char   g_output_dir[512]   = "data/synthetic/transactions";
static char   g_stores_csv[512]   = "data/real/stores.csv";

/* Real store IDs loaded from CSV (sorted for geographic proximity) */
static int   *g_store_ids;

#define NUM_STATES    50
#define WRITE_BUF_SZ  (16 * 1024 * 1024)  /* 16 MB */

/* ========================================================================
 * xoshiro256** PRNG — fast, high-quality, one instance per thread
 * ======================================================================== */

typedef struct { uint64_t s[4]; } rng_t;

static inline uint64_t rotl64(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static inline uint64_t rng_next(rng_t *r) {
    const uint64_t result = rotl64(r->s[1] * 5, 7) * 9;
    const uint64_t t = r->s[1] << 17;
    r->s[2] ^= r->s[0];  r->s[3] ^= r->s[1];
    r->s[1] ^= r->s[2];  r->s[0] ^= r->s[3];
    r->s[2] ^= t;
    r->s[3] = rotl64(r->s[3], 45);
    return result;
}

static void rng_seed(rng_t *r, uint64_t seed) {
    for (int i = 0; i < 4; i++) {
        seed += 0x9e3779b97f4a7c15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        r->s[i] = z ^ (z >> 31);
    }
}

static inline double rng_double(rng_t *r) {
    return (double)(rng_next(r) >> 11) / (double)(1ULL << 53);
}

/* ========================================================================
 * Date table: 2024-01-01 to 2025-12-31 (731 days)
 * ======================================================================== */

typedef struct {
    int16_t year, month, day;   /* 1-based */
    int16_t dow;                /* 0=Mon … 6=Sun */
} date_info_t;

#define MAX_DAYS 732
static date_info_t g_dates[MAX_DAYS];
static int g_total_days;

static const int dm24[] = {31,29,31,30,31,30,31,31,30,31,30,31};
static const int dm25[] = {31,28,31,30,31,30,31,31,30,31,30,31};

static void init_dates(void) {
    int idx = 0, dow = 0; /* 2024-01-01 = Monday */
    for (int m = 0; m < 12; m++)
        for (int d = 1; d <= dm24[m]; d++, idx++) {
            g_dates[idx] = (date_info_t){2024, m+1, d, dow};
            dow = (dow + 1) % 7;
        }
    for (int m = 0; m < 12; m++)
        for (int d = 1; d <= dm25[m]; d++, idx++) {
            g_dates[idx] = (date_info_t){2025, m+1, d, dow};
            dow = (dow + 1) % 7;
        }
    g_total_days = idx;
}

/* ========================================================================
 * Product prices — deterministic from seed
 *   15%: $2-5,  30%: $5-10,  30%: $10-20,  17%: $20-35,  8%: $35-50
 * ======================================================================== */

static double *g_product_prices;

static void init_product_prices(void) {
    g_product_prices = malloc(g_num_products * sizeof(double));
    rng_t rng;
    rng_seed(&rng, 12345);
    for (int i = 0; i < g_num_products; i++) {
        double u = rng_double(&rng);
        double p;
        if      (u < 0.15) p =  2.0 + rng_double(&rng) *  3.0;
        else if (u < 0.45) p =  5.0 + rng_double(&rng) *  5.0;
        else if (u < 0.75) p = 10.0 + rng_double(&rng) * 10.0;
        else if (u < 0.92) p = 20.0 + rng_double(&rng) * 15.0;
        else               p = 35.0 + rng_double(&rng) * 15.0;
        g_product_prices[i] = floor(p * 100.0 + 0.5) / 100.0;
    }
}

/* ========================================================================
 * Zipf CDFs with seasonal adjustments (12 calendar months)
 *   weight[i] = 1/(i+1)^1.07, with seasonal multipliers
 * ======================================================================== */

static double *g_product_cdfs;  /* [12][num_products] */

static void init_product_cdfs(void) {
    int np = g_num_products;
    g_product_cdfs = malloc(12 * np * sizeof(double));
    double *base = malloc(np * sizeof(double));

    for (int i = 0; i < np; i++)
        base[i] = 1.0 / pow((double)(i + 1), 1.07);

    for (int m = 0; m < 12; m++) {
        double *cdf = g_product_cdfs + m * np;
        int cal = m + 1;
        double total = 0.0;

        for (int i = 0; i < np; i++) {
            double w = base[i];
            /* Cold/flu (0-999): boost Dec-Feb, suppress Jun-Aug */
            if (i < 1000) {
                if (cal == 12 || cal <= 2)     w *= 2.5;
                else if (cal >= 6 && cal <= 8) w *= 0.6;
            }
            /* Skin/sunscreen (1000-1499): boost Jun-Aug */
            else if (i < 1500) {
                if (cal >= 6 && cal <= 8)      w *= 1.8;
            }
            /* Greeting cards (6000-6999): spike Dec, Feb */
            else if (i >= 6000 && i < 7000) {
                if (cal == 12)                 w *= 2.5;
                else if (cal == 2)             w *= 2.0;
            }
            cdf[i] = w;
            total += w;
        }
        double cum = 0.0;
        for (int i = 0; i < np; i++) {
            cum += cdf[i] / total;
            cdf[i] = cum;
        }
        cdf[np - 1] = 1.0;
    }
    free(base);
}

static inline int sample_product(rng_t *r, int cal_m0) {
    double u = rng_double(r);
    double *cdf = g_product_cdfs + cal_m0 * g_num_products;
    int lo = 0, hi = g_num_products - 1;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (cdf[mid] < u) lo = mid + 1; else hi = mid;
    }
    return lo;
}

/* ========================================================================
 * State sales tax (customer_id % 50)
 * ======================================================================== */

static const double g_state_tax[NUM_STATES] = {
    0.0400, 0.0000, 0.0560, 0.0650, 0.0725, 0.0290, 0.0635, 0.0000, 0.0600, 0.0400,
    0.0400, 0.0600, 0.0625, 0.0700, 0.0600, 0.0650, 0.0600, 0.0445, 0.0550, 0.0600,
    0.0625, 0.0600, 0.0688, 0.0700, 0.0423, 0.0000, 0.0550, 0.0685, 0.0000, 0.0663,
    0.0513, 0.0400, 0.0475, 0.0500, 0.0575, 0.0450, 0.0000, 0.0600, 0.0700, 0.0600,
    0.0450, 0.0700, 0.0625, 0.0610, 0.0600, 0.0530, 0.0650, 0.0600, 0.0500, 0.0400,
};

/* ========================================================================
 * Hour-of-day (7-21)
 * ======================================================================== */

#define HOUR_MIN  7
#define HOUR_SLOTS 15

static const int g_wd_wt[HOUR_SLOTS] = {2,4,6,8,12,14,12,8,7,8,12,14,10,6,3};
static const int g_we_wt[HOUR_SLOTS] = {2,4,8,14,16,16,14,10,8,7,6,5,4,3,2};
static int g_wd_total, g_we_total;

static void init_hour_weights(void) {
    g_wd_total = g_we_total = 0;
    for (int i = 0; i < HOUR_SLOTS; i++) {
        g_wd_total += g_wd_wt[i];
        g_we_total += g_we_wt[i];
    }
}

static inline int sample_hour(rng_t *r, int is_weekend) {
    const int *w = is_weekend ? g_we_wt : g_wd_wt;
    int v = (int)(rng_next(r) % (is_weekend ? g_we_total : g_wd_total));
    int c = 0;
    for (int i = 0; i < HOUR_SLOTS; i++) {
        c += w[i];
        if (v < c) return HOUR_MIN + i;
    }
    return HOUR_MIN + HOUR_SLOTS - 1;
}

/* ========================================================================
 * Store IDs — load real CVS store numbers from CSV
 * ======================================================================== */

static int compare_int(const void *a, const void *b) {
    return *(const int *)a - *(const int *)b;
}

static void init_store_ids(void) {
    FILE *fp = fopen(g_stores_csv, "r");
    if (!fp) {
        fprintf(stderr, "WARNING: Cannot open %s — using sequential store IDs\n",
                g_stores_csv);
        g_store_ids = malloc(g_num_stores * sizeof(int));
        for (int i = 0; i < g_num_stores; i++)
            g_store_ids[i] = i + 1;
        return;
    }

    /* First pass: count data lines */
    char line[4096];
    int count = 0;
    if (!fgets(line, sizeof(line), fp)) { fclose(fp); return; } /* skip header */
    while (fgets(line, sizeof(line), fp))
        if (line[0] != '\n' && line[0] != '\0') count++;

    /* Second pass: read store_id (first CSV field) */
    rewind(fp);
    fgets(line, sizeof(line), fp); /* skip header */

    g_store_ids = malloc(count * sizeof(int));
    int idx = 0;
    while (fgets(line, sizeof(line), fp) && idx < count) {
        int id = atoi(line);  /* atoi stops at first non-digit (the comma) */
        if (id > 0)
            g_store_ids[idx++] = id;
    }
    fclose(fp);

    g_num_stores = idx;

    /* Sort so adjacent indices approximate geographic proximity */
    qsort(g_store_ids, g_num_stores, sizeof(int), compare_int);

    fprintf(stderr, "Store IDs: loaded %d real store IDs from %s (range %d–%d)\n",
            g_num_stores, g_stores_csv, g_store_ids[0], g_store_ids[g_num_stores - 1]);
}

/* ========================================================================
 * Sampling helpers
 * ======================================================================== */

/* Quantity: 1 (82%), 2 (15%), 3 (3%) */
static inline int sample_quantity(rng_t *r) {
    int v = (int)(rng_next(r) % 100);
    return (v < 82) ? 1 : (v < 97) ? 2 : 3;
}

/* Discount: 30% none, 40% 5-15%, 20% 15-30%, 10% 30-50% */
static inline double sample_discount(rng_t *r) {
    int v = (int)(rng_next(r) % 100);
    if (v < 30) return 0.0;
    double u = rng_double(r);
    if (v < 70) return 0.05 + u * 0.10;
    if (v < 90) return 0.15 + u * 0.15;
    return 0.30 + u * 0.20;
}

/* Triangular distribution int in [min, max] with mode */
static inline int triangular_int(rng_t *r, int mn, int mode, int mx) {
    double u = rng_double(r);
    double fc = (double)(mode - mn) / (double)(mx - mn);
    double x;
    if (u < fc)
        x = mn + sqrt(u * (mx - mn) * (mode - mn));
    else
        x = mx - sqrt((1.0 - u) * (mx - mn) * (mx - mode));
    int res = (int)(x + 0.5);
    if (res < mn) res = mn;
    if (res > mx) res = mx;
    return res;
}

/* ========================================================================
 * Compression: zstd > gzip > raw fallback
 * ======================================================================== */

enum { COMP_ZSTD, COMP_GZIP, COMP_RAW };
static int g_comp = COMP_ZSTD;

static void detect_compression(void) {
    if (system("command -v zstd > /dev/null 2>&1") == 0) {
        g_comp = COMP_ZSTD;
        fprintf(stderr, "Compression: zstd level 1\n");
    } else if (system("command -v gzip > /dev/null 2>&1") == 0) {
        g_comp = COMP_GZIP;
        fprintf(stderr, "Compression: gzip (fallback)\n");
    } else {
        g_comp = COMP_RAW;
        fprintf(stderr, "WARNING: No zstd/gzip — raw CSV (~1.5TB for full run)\n");
    }
}

static const char *file_ext(void) {
    return (g_comp == COMP_ZSTD) ? ".csv.zst" :
           (g_comp == COMP_GZIP) ? ".csv.gz"  : ".csv";
}

static FILE *open_out(const char *path) {
    char cmd[1024];
    if (g_comp == COMP_ZSTD) {
        snprintf(cmd, sizeof(cmd), "zstd -1 -q -f -o \"%s\"", path);
        return popen(cmd, "w");
    } else if (g_comp == COMP_GZIP) {
        snprintf(cmd, sizeof(cmd), "gzip > \"%s\"", path);
        return popen(cmd, "w");
    }
    return fopen(path, "w");
}

static void close_out(FILE *fp) {
    if (!fp) return;
    if (g_comp == COMP_RAW) fclose(fp); else pclose(fp);
}

/* ========================================================================
 * Thread worker — one output file per thread
 * ======================================================================== */

typedef struct {
    int      thread_id;
    int      cust_start;   /* 0-based */
    int      cust_end;     /* exclusive */
    uint64_t rows_written;
} thread_arg_t;

static const char *HDR =
    "transaction_id,loyalty_number,customer_id,store_id,product_id,"
    "quantity,unit_price,discount_pct,discount_amt,subtotal,"
    "tax_rate,tax_amt,total,date,hour\n";

static void *generate_chunk(void *arg) {
    thread_arg_t *ta = (thread_arg_t *)arg;
    int tid = ta->thread_id;

    rng_t rng;
    rng_seed(&rng, (uint64_t)tid * 2654435761ULL + 42);

    /* One output file per thread */
    char path[1024];
    snprintf(path, sizeof(path), "%s/txns_t%02d%s", g_output_dir, tid, file_ext());
    FILE *fp = open_out(path);
    if (!fp) {
        fprintf(stderr, "ERROR: thread %d cannot open %s\n", tid, path);
        ta->rows_written = 0;
        return NULL;
    }
    char *wbuf = malloc(WRITE_BUF_SZ);
    if (wbuf) setvbuf(fp, wbuf, _IOFBF, WRITE_BUF_SZ);
    fprintf(fp, "%s", HDR);

    uint64_t rows = 0;

    for (int c = ta->cust_start; c < ta->cust_end; c++) {
        int cid = c + 1;
        int home_idx = cid % g_num_stores;
        double tax_rate = g_state_tax[cid % NUM_STATES];

        int items = 0;
        int day = (int)(rng_next(&rng) % 30);

        while (items < g_txns_per_customer) {
            if (day >= g_total_days) day %= g_total_days;
            date_info_t *d = &g_dates[day];
            int cal_m0     = d->month - 1;
            int is_we      = (d->dow >= 5);

            int basket = triangular_int(&rng, 1, 4, 12);
            int left   = g_txns_per_customer - items;
            if (basket > left) basket = left;

            /* Store: 90% home, 10% nearby ±50 (index offset in sorted store array) */
            int store_idx;
            if (rng_double(&rng) < 0.90) {
                store_idx = home_idx;
            } else {
                store_idx = home_idx + (int)(rng_next(&rng) % 101) - 50;
                if (store_idx < 0) store_idx = 0;
                if (store_idx >= g_num_stores) store_idx = g_num_stores - 1;
            }
            int store = g_store_ids[store_idx];

            int hour = sample_hour(&rng, is_we);

            for (int b = 0; b < basket; b++) {
                int pidx = sample_product(&rng, cal_m0);
                int qty  = sample_quantity(&rng);
                double up   = g_product_prices[pidx];
                double dpct = sample_discount(&rng);
                double damt = up * qty * dpct;
                double sub  = up * qty - damt;
                double tamt = sub * tax_rate;
                double tot  = sub + tamt;
                uint64_t txn_id = (uint64_t)c * g_txns_per_customer + items + 1;

                fprintf(fp,
                    "%" PRIu64 ",EC%010d,%d,%d,%d,"
                    "%d,%.2f,%.4f,%.2f,%.2f,"
                    "%.4f,%.2f,%.2f,%04d-%02d-%02d,%d\n",
                    txn_id, cid, cid, store, pidx + 1,
                    qty, up, dpct, damt, sub,
                    tax_rate, tamt, tot,
                    d->year, d->month, d->day, hour);

                items++;
            }
            rows += basket;
            day += 2 + (int)(rng_next(&rng) % 3);
        }

        /* Progress every 10,000 customers */
        int done = c - ta->cust_start + 1;
        int total_c = ta->cust_end - ta->cust_start;
        if (done % 10000 == 0)
            fprintf(stderr, "  Thread %d: %d / %d (%.0f%%)\n",
                    tid, done, total_c, 100.0 * done / total_c);
    }

    close_out(fp);
    free(wbuf);
    ta->rows_written = rows;
    return NULL;
}

/* ========================================================================
 * Test mode: print sample rows
 * ======================================================================== */

static void print_sample_rows(void) {
    char path[1024], cmd[1100];
    snprintf(path, sizeof(path), "%s/txns_t00%s", g_output_dir, file_ext());

    if (g_comp == COMP_ZSTD)
        snprintf(cmd, sizeof(cmd), "zstd -d -c \"%s\" 2>/dev/null | head -6", path);
    else if (g_comp == COMP_GZIP)
        snprintf(cmd, sizeof(cmd), "gzip -d -c \"%s\" 2>/dev/null | head -6", path);
    else
        snprintf(cmd, sizeof(cmd), "head -6 \"%s\"", path);

    printf("\n=== Sample rows from %s ===\n", path);
    fflush(stdout);
    system(cmd);
}

/* ========================================================================
 * Main
 * ======================================================================== */

int main(int argc, char *argv[]) {
    if (argc >= 2) g_num_customers     = atoi(argv[1]);
    if (argc >= 3) g_txns_per_customer = atoi(argv[2]);
    if (argc >= 4) g_num_products      = atoi(argv[3]);
    /* arg 4 (num_stores) is ignored — real count comes from stores CSV */
    if (argc >= 6) strncpy(g_output_dir, argv[5], sizeof(g_output_dir) - 1);
    if (argc >= 7) strncpy(g_stores_csv, argv[6], sizeof(g_stores_csv) - 1);

    int is_test = (g_num_customers < 10000);

    long long total_expected = (long long)g_num_customers * g_txns_per_customer;
    printf("txn_generator\n");
    printf("  Customers:          %d\n", g_num_customers);
    printf("  Txns/customer:      %d\n", g_txns_per_customer);
    printf("  Total transactions: %lld\n", total_expected);
    printf("  Products:           %d\n", g_num_products);
    printf("  Stores:             %d\n", g_num_stores);
    printf("  Threads:            %d\n", g_num_threads);
    printf("  Output:             %s/\n\n", g_output_dir);
    fflush(stdout);

    /* Create output directory */
    char cmd[600];
    snprintf(cmd, sizeof(cmd), "mkdir -p \"%s\"", g_output_dir);
    system(cmd);

    detect_compression();
    init_dates();
    init_store_ids();
    init_product_prices();
    init_product_cdfs();
    init_hour_weights();

    fprintf(stderr, "Product prices: %d products initialized\n", g_num_products);
    fprintf(stderr, "Zipf CDFs: 12 monthly variants (s=1.07)\n");
    fprintf(stderr, "Date table: %d days\n", g_total_days);
    fprintf(stderr, "Launching %d threads...\n\n", g_num_threads);

    struct timespec ts0, ts1;
    clock_gettime(CLOCK_MONOTONIC, &ts0);

    pthread_t threads[8];
    thread_arg_t args[8];
    int per = g_num_customers / g_num_threads;
    int rem = g_num_customers % g_num_threads;

    for (int i = 0; i < g_num_threads; i++) {
        args[i].thread_id   = i;
        args[i].cust_start  = i * per + (i < rem ? i : rem);
        args[i].cust_end    = args[i].cust_start + per + (i < rem ? 1 : 0);
        args[i].rows_written = 0;
        pthread_create(&threads[i], NULL, generate_chunk, &args[i]);
    }

    for (int i = 0; i < g_num_threads; i++)
        pthread_join(threads[i], NULL);

    clock_gettime(CLOCK_MONOTONIC, &ts1);

    uint64_t total_rows = 0;
    for (int i = 0; i < g_num_threads; i++)
        total_rows += args[i].rows_written;

    double elapsed = (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) / 1e9;

    printf("\n========================================\n");
    printf("  Total rows:  %" PRIu64 "\n", total_rows);
    printf("  Elapsed:     %.1f seconds\n", elapsed);
    printf("  Throughput:  %.2f M rows/sec\n", (double)total_rows / elapsed / 1e6);
    printf("  Output:      %d files\n", g_num_threads);
    printf("========================================\n");
    fflush(stdout);

    if (is_test)
        print_sample_rows();

    free(g_product_prices);
    free(g_product_cdfs);
    free(g_store_ids);
    return 0;
}
