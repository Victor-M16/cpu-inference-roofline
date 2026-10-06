// roofline_probe: measure the two roofs of a CPU and the empirical curve between them.
//
//   1. Memory roof: sustained DRAM bandwidth (STREAM-style triad, plus a read-only
//      reduction, which is the closer analogue of streaming weights during decode).
//   2. Compute roof: peak fp32 FMA throughput with enough independent accumulators
//      to hide FMA latency.
//   3. Intensity sweep: stream a DRAM-sized array and do K FMAs per loaded vector, for
//      K = 1..256. Plotting GFLOP/s against FLOP/byte traces the roofline measured,
//      not assumed: flat bandwidth slope on the left, compute ceiling on the right.
//
// Portable C11 + OpenMP + GCC/Clang vector extensions, so it builds unchanged on
// x86 (SSE/AVX2/AVX-512) and Arm (NEON). Build with -O3 -march=native -fopenmp.
// Prints one JSON object to stdout.

#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__AVX512F__)
#define VBYTES 64
#define ISA "avx512"
#elif defined(__AVX__)
#define VBYTES 32
#define ISA "avx2"
#elif defined(__ARM_NEON) || defined(__SSE2__)
#define VBYTES 16
#if defined(__ARM_NEON)
#define ISA "neon"
#else
#define ISA "sse2"
#endif
#else
#define VBYTES 16
#define ISA "generic"
#endif

typedef float vf __attribute__((vector_size(VBYTES)));
#define LANES (VBYTES / (int)sizeof(float))

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static size_t g_n;     // floats per array
static int g_reps = 5; // best-of

static float *alloc_floats(size_t n) {
    float *p = NULL;
    if (posix_memalign((void **)&p, 4096, n * sizeof(float)) != 0) {
        fprintf(stderr, "allocation of %zu MiB failed\n", (n * sizeof(float)) >> 20);
        exit(1);
    }
    return p;
}

// ---------------------------------------------------------------- memory roof

static double bw_triad(float *a, const float *b, const float *c, int threads) {
    double best = 1e30;
    const float s = 1.0001f;
    for (int r = 0; r < g_reps; r++) {
        double t0 = now();
#pragma omp parallel for num_threads(threads) schedule(static)
        for (size_t i = 0; i < g_n; i++) a[i] = b[i] + s * c[i];
        double dt = now() - t0;
        if (dt < best) best = dt;
    }
    return 3.0 * g_n * sizeof(float) / best / 1e9; // STREAM convention: no write-allocate
}

static volatile float g_sink;

static double bw_read(const float *a, int threads) {
    double best = 1e30;
    const size_t nv = g_n / LANES;
    const vf *av = (const vf *)a;
    for (int r = 0; r < g_reps; r++) {
        float total = 0;
        double t0 = now();
#pragma omp parallel num_threads(threads) reduction(+ : total)
        {
            vf s0 = {0}, s1 = {0}, s2 = {0}, s3 = {0};
#pragma omp for schedule(static)
            for (size_t i = 0; i < nv - 3; i += 4) {
                s0 += av[i];
                s1 += av[i + 1];
                s2 += av[i + 2];
                s3 += av[i + 3];
            }
            vf s = s0 + s1 + s2 + s3;
            for (int l = 0; l < LANES; l++) total += s[l];
        }
        double dt = now() - t0;
        g_sink = total;
        if (dt < best) best = dt;
    }
    return (double)g_n * sizeof(float) / best / 1e9;
}

// ---------------------------------------------------------------- compute roof

#define NACC 16
static double peak_flops(int threads) {
    const long iters = 20 * 1000 * 1000;
    double best = 1e30;
    for (int r = 0; r < 3; r++) {
        float total = 0;
        double t0 = now();
#pragma omp parallel num_threads(threads) reduction(+ : total)
        {
            vf acc[NACC];
            vf m, k;
            for (int l = 0; l < LANES; l++) {
                m[l] = 0.999999f;
                k[l] = 1e-7f * (omp_get_thread_num() + 1);
            }
            for (int j = 0; j < NACC; j++) acc[j] = k * (float)j;
            for (long i = 0; i < iters; i++) {
#pragma GCC unroll 16
                for (int j = 0; j < NACC; j++) acc[j] = acc[j] * m + k;
            }
            vf s = {0};
            for (int j = 0; j < NACC; j++) s += acc[j];
            for (int l = 0; l < LANES; l++) total += s[l];
        }
        double dt = now() - t0;
        g_sink = total;
        if (dt < best) best = dt;
    }
    return 2.0 * LANES * NACC * (double)iters * threads / best / 1e9;
}

// ---------------------------------------------------------------- intensity sweep
// For each vector loaded from DRAM, do K FMAs spread over 8 independent chains.
// FLOPs per byte = 2*K*LANES / VBYTES = K/2 for fp32.

#define SWEEP_FN(K)                                                                      \
    static double sweep_##K(const float *a, int threads) {                               \
        const size_t nv = g_n / LANES;                                                   \
        const vf *av = (const vf *)a;                                                    \
        double best = 1e30;                                                              \
        for (int r = 0; r < 3; r++) {                                                    \
            float total = 0;                                                             \
            double t0 = now();                                                           \
            _Pragma("omp parallel num_threads(threads) reduction(+ : total)") {          \
                vf c[8];                                                                 \
                for (int j = 0; j < 8; j++)                                              \
                    for (int l = 0; l < LANES; l++) c[j][l] = 1e-6f * (j + 1);           \
                _Pragma("omp for schedule(static)") for (size_t i = 0; i < nv; i++) {    \
                    vf v = av[i];                                                        \
                    _Pragma("GCC unroll 256") for (int q = 0; q < K; q++) c[q & 7] =     \
                        c[q & 7] * v + c[(q + 1) & 7];                                   \
                }                                                                        \
                vf s = {0};                                                              \
                for (int j = 0; j < 8; j++) s += c[j];                                   \
                for (int l = 0; l < LANES; l++) total += s[l];                           \
            }                                                                            \
            double dt = now() - t0;                                                      \
            g_sink = total;                                                              \
            if (dt < best) best = dt;                                                    \
        }                                                                                \
        return 2.0 * K * (double)g_n / best / 1e9;                                       \
    }

SWEEP_FN(1)
SWEEP_FN(2)
SWEEP_FN(4)
SWEEP_FN(8)
SWEEP_FN(16)
SWEEP_FN(32)
SWEEP_FN(64)
SWEEP_FN(128)
SWEEP_FN(256)

typedef double (*sweep_fn)(const float *, int);
static const int sweep_k[] = {1, 2, 4, 8, 16, 32, 64, 128, 256};
static const sweep_fn sweep_fns[] = {sweep_1,  sweep_2,  sweep_4,   sweep_8,  sweep_16,
                                     sweep_32, sweep_64, sweep_128, sweep_256};

int main(int argc, char **argv) {
    size_t mib = 256; // per array; three arrays for triad
    int threads = omp_get_max_threads();
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--mib") && i + 1 < argc) mib = strtoul(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else {
            fprintf(stderr, "usage: %s [--mib N_per_array] [--threads T]\n", argv[0]);
            return 2;
        }
    }
    g_n = mib * 1024 * 1024 / sizeof(float);
    g_n -= g_n % (LANES * 4);

    float *a = alloc_floats(g_n), *b = alloc_floats(g_n), *c = alloc_floats(g_n);
#pragma omp parallel for num_threads(threads) schedule(static)
    for (size_t i = 0; i < g_n; i++) { // first touch on the threads that will use it
        a[i] = 0.f;
        b[i] = 1.f;
        c[i] = 2.f;
    }

    printf("{\n  \"isa\": \"%s\",\n  \"vector_bytes\": %d,\n  \"threads\": %d,\n"
           "  \"array_mib\": %zu,\n",
           ISA, VBYTES, threads, mib);
    fflush(stdout);

    // Thread scaling for both roofs: decode tracks the first, prefill the second.
    printf("  \"scaling\": [\n");
    for (int t = 1; t <= threads; t = (t * 2 > threads && t != threads) ? threads : t * 2) {
        double rd = bw_read(a, t), tr = bw_triad(a, b, c, t), pf = peak_flops(t);
        printf("    {\"threads\": %d, \"read_gbs\": %.2f, \"triad_gbs\": %.2f, "
               "\"peak_gflops\": %.2f}%s\n",
               t, rd, tr, pf, t == threads ? "" : ",");
        fflush(stdout);
    }
    printf("  ],\n");

    printf("  \"sweep\": [\n");
    int ns = sizeof(sweep_k) / sizeof(sweep_k[0]);
    for (int s = 0; s < ns; s++) {
        double g = sweep_fns[s](a, threads);
        printf("    {\"flops_per_byte\": %.3f, \"gflops\": %.2f}%s\n", sweep_k[s] / 2.0, g,
               s == ns - 1 ? "" : ",");
        fflush(stdout);
    }
    printf("  ]\n}\n");

    free(a);
    free(b);
    free(c);
    return 0;
}
