/* The Vidya forward in plain C, one byte at a time, as zara_poles.Sampler computes it:
 * polar poles, dense projections, ternary weights (2-bit codes and one scale a matrix) or float
 * weights, and optionally int8 projection inputs (BitNet b1.58 absmax per position). Model files
 * come from export_ternary.py.
 *
 *   cc -O3 -march=native -fopenmp -o vidya_ternary vidya_ternary.c -lm
 *   vidya_ternary MODEL.bin check REF.bin     # logits and bits per byte against the PyTorch reference
 *   vidya_ternary MODEL.bin bench N           # bytes a second over N bytes of the forward
 *   vidya_ternary MODEL.bin eval HELDOUT K    # bits per byte over K windows of 2048 held-out bytes
 *
 * Weights are float, ternary (2-bit codes, one scale a matrix) or int8 (one scale a row, the usual
 * post-training quantization); eval scores every model the same way, so they compare directly.
 *
 * With ternary weights and int8 inputs a projection row is a sum of +q, -q or nothing over the
 * row's codes (an int32 accumulator) and one float multiply (the weight scale times the input
 * scale). The recurrence, the norms, the spatial rotation and the head are float. Threads split
 * the rows of each projection and of the head (OMP_NUM_THREADS).
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define EPS_LN 1e-5f
#define EPS_NORMALIZE 1e-12f

typedef struct {
    int rows, cols;
    float scale;          /* ternary: the absmean scale */
    const float *row_scale; /* int8: [rows] */
    int8_t *code;         /* ternary: [rows][cols] in {-1, 0, 1}; int8: [rows][cols] */
    const float *w;       /* float: [rows][cols] */
    const float *bias;
} Proj;

typedef struct {
    const float *ln_w, *ln_b, *qh_w, *kh_w, *spec, *cre, *cim, *cosm1, *sinv, *coupling, *gate;
    Proj qkv, out;
    float *sre, *sim;     /* the state [H][L][dh], real and imaginary */
} Block;

typedef struct {
    int D, H, L, dh, layers, G, W, F, ternary, act8;
    Block *b;
    const float *lnf_w, *lnf_b, *head, *table;
    float *sum;           /* the running sum of ln_f outputs [G][W] */
    long count;
    /* scratch */
    float *z, *u, *qkv, *qh, *kh, *read, *temporal, *n, *f, *feat, *logits, *xq;
    int8_t *q8;
} Model;

static const uint8_t *cursor, *end;

static const void *take(size_t bytes) {
    if (cursor + bytes > end) { fprintf(stderr, "model file too short\n"); exit(1); }
    const void *p = cursor;
    cursor += bytes;
    return p;
}
static const float *floats(size_t n) { return (const float *)take(n * sizeof(float)); }

static Proj projection(Model *m, int rows, int cols) {
    Proj p = {rows, cols, 1.0f, NULL, NULL, NULL, NULL};
    size_t n = (size_t)rows * cols;
    if (m->ternary == 2) {
        p.row_scale = floats(rows);
        p.code = (int8_t *)take(n);
    } else if (m->ternary) {
        p.scale = *floats(1);
        const uint8_t *packed = (const uint8_t *)take((n + 3) / 4);
        p.code = malloc(n);
        for (size_t i = 0; i < n; i++) {
            int c = (packed[i / 4] >> (2 * (i % 4))) & 3;
            p.code[i] = c == 1 ? 1 : c == 2 ? -1 : 0;
        }
    } else {
        p.w = floats(n);
    }
    p.bias = floats(rows);
    return p;
}

static void load(Model *m, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *buf = malloc(size);
    if (fread(buf, 1, size, f) != (size_t)size) { fprintf(stderr, "read failed\n"); exit(1); }
    fclose(f);
    cursor = buf;
    end = buf + size;
    if (memcmp(take(4), "VTRN", 4)) { fprintf(stderr, "not a VTRN model\n"); exit(1); }
    const int32_t *h = (const int32_t *)take(11 * 4);
    if (h[0] != 1) { fprintf(stderr, "model version %d, expected 1\n", h[0]); exit(1); }
    m->D = h[1]; m->H = h[2]; m->L = h[3]; m->dh = h[4]; m->layers = h[5]; m->G = h[6]; m->W = h[7];
    m->F = h[8]; m->ternary = h[9]; m->act8 = h[10];
    int D = m->D, H = m->H, L = m->L, dh = m->dh, G = m->G, W = m->W;
    m->b = calloc(m->layers, sizeof(Block));
    for (int i = 0; i < m->layers; i++) {
        Block *b = &m->b[i];
        b->ln_w = floats(D); b->ln_b = floats(D);
        b->qkv = projection(m, 3 * D, D);
        b->qh_w = floats((size_t)H * 2 * L * dh); b->kh_w = floats((size_t)H * L * dh); b->spec = floats(H * L);
        b->cre = floats(H * L); b->cim = floats(H * L);
        b->out = projection(m, D, D);
        b->cosm1 = floats((G / 2) * W); b->sinv = floats((G / 2) * W);
        b->coupling = floats(G); b->gate = floats(G);
        b->sre = calloc((size_t)H * L * dh, sizeof(float));
        b->sim = calloc((size_t)H * L * dh, sizeof(float));
    }
    m->lnf_w = floats(D); m->lnf_b = floats(D); m->head = floats((size_t)256 * m->F); m->table = floats((size_t)256 * D);
    if (cursor != end) { fprintf(stderr, "%ld bytes left over in the model file\n", (long)(end - cursor)); exit(1); }
    m->sum = calloc(G * W, sizeof(float));
    m->count = 0;
    m->z = malloc(D * 4); m->u = malloc(D * 4); m->qkv = malloc(3 * D * 4); m->qh = malloc(H * 2 * L * 4);
    m->kh = malloc(H * L * 4); m->read = malloc(D * 4); m->temporal = malloc(D * 4); m->n = malloc(D * 4);
    m->f = malloc(D * 4); m->feat = malloc(m->F * 4); m->logits = malloc(256 * 4); m->xq = malloc(D * 4);
    m->q8 = malloc(D);
}

static void reset(Model *m) {
    for (int i = 0; i < m->layers; i++) {
        memset(m->b[i].sre, 0, (size_t)m->H * m->L * m->dh * 4);
        memset(m->b[i].sim, 0, (size_t)m->H * m->L * m->dh * 4);
    }
    memset(m->sum, 0, m->G * m->W * 4);
    m->count = 0;
}

static void layer_norm(const float *x, const float *w, const float *b, float *y, int n) {
    float mean = 0, var = 0;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;
    for (int i = 0; i < n; i++) var += (x[i] - mean) * (x[i] - mean);
    var /= n;
    float inv = 1.0f / sqrtf(var + EPS_LN);
    for (int i = 0; i < n; i++) y[i] = (x[i] - mean) * inv * w[i] + b[i];
}

static float norm(const float *x, int n) {
    float s = 0;
    for (int i = 0; i < n; i++) s += x[i] * x[i];
    return sqrtf(s);
}

/* y = W x + bias, with x rounded to int8 per call when act8 (zara_poles.act8) */
static void apply(Model *m, const Proj *p, const float *x, float *y) {
    int rows = p->rows, cols = p->cols;
    float deq = 1.0f;
    if (m->act8) {
        float s = 0;
        for (int j = 0; j < cols; j++) s = fmaxf(s, fabsf(x[j]));
        s = fmaxf(s, 1e-5f);
        float inv = 127.0f / s;
        deq = s / 127.0f;
        for (int j = 0; j < cols; j++) {
            float r = nearbyintf(x[j] * inv);   /* round half to even, as torch.round */
            r = fminf(fmaxf(r, -128.0f), 127.0f);
            m->q8[j] = (int8_t)r;
            m->xq[j] = r * deq;
        }
        x = m->xq;
    }
    if (m->ternary == 2) {
        if (m->act8) {
            const int8_t *q8 = m->q8;
            #pragma omp parallel for schedule(static)
            for (int r = 0; r < rows; r++) {
                const int8_t *c = p->code + (size_t)r * cols;
                int32_t acc = 0;
                for (int j = 0; j < cols; j++) acc += c[j] * q8[j];
                y[r] = (float)acc * (p->row_scale[r] * deq) + p->bias[r];
            }
        } else {
            #pragma omp parallel for schedule(static)
            for (int r = 0; r < rows; r++) {
                const int8_t *c = p->code + (size_t)r * cols;
                float acc = 0;
                for (int j = 0; j < cols; j++) acc += c[j] * x[j];
                y[r] = acc * p->row_scale[r] + p->bias[r];
            }
        }
    } else if (m->ternary && m->act8) {
        const int8_t *q8 = m->q8;
        float k = p->scale * deq;
        #pragma omp parallel for schedule(static)
        for (int r = 0; r < rows; r++) {
            const int8_t *c = p->code + (size_t)r * cols;
            int32_t acc = 0;
            for (int j = 0; j < cols; j++) acc += c[j] * q8[j];
            y[r] = (float)acc * k + p->bias[r];
        }
    } else if (m->ternary) {
        #pragma omp parallel for schedule(static)
        for (int r = 0; r < rows; r++) {
            const int8_t *c = p->code + (size_t)r * cols;
            float acc = 0;
            for (int j = 0; j < cols; j++) acc += c[j] * x[j];
            y[r] = acc * p->scale + p->bias[r];
        }
    } else {
        #pragma omp parallel for schedule(static)
        for (int r = 0; r < rows; r++) {
            const float *w = p->w + (size_t)r * cols;
            float acc = 0;
            for (int j = 0; j < cols; j++) acc += w[j] * x[j];
            y[r] = acc + p->bias[r];
        }
    }
}

/* the logits after one byte (Sampler.logits) */
static const float *step(Model *m, int byte) {
    int D = m->D, H = m->H, L = m->L, dh = m->dh, G = m->G, W = m->W;
    float *z = m->z;
    memcpy(z, m->table + (size_t)byte * D, D * 4);
    for (int i = 0; i < m->layers; i++) {
        Block *b = &m->b[i];
        layer_norm(z, b->ln_w, b->ln_b, m->u, D);
        apply(m, &b->qkv, m->u, m->qkv);
        const float *q = m->qkv, *k = m->qkv + D, *v = m->qkv + 2 * D;
        for (int h = 0; h < H; h++) {
            for (int l = 0; l < 2 * L; l++) {
                const float *w = b->qh_w + ((size_t)h * 2 * L + l) * dh;
                float s = 0;
                for (int d = 0; d < dh; d++) s += q[h * dh + d] * w[d];
                m->qh[h * 2 * L + l] = s;
            }
            for (int l = 0; l < L; l++) {
                const float *w = b->kh_w + ((size_t)h * L + l) * dh;
                float s = 0;
                for (int d = 0; d < dh; d++) s += k[h * dh + d] * w[d];
                m->kh[h * L + l] = s * b->spec[h * L + l];
            }
        }
        for (int h = 0; h < H; h++) {
            float *rd = m->read + h * dh;
            memset(rd, 0, dh * 4);
            for (int l = 0; l < L; l++) {
                float cr = b->cre[h * L + l], ci = b->cim[h * L + l], kk = m->kh[h * L + l];
                float qr = m->qh[h * 2 * L + l], qi = m->qh[h * 2 * L + L + l];
                float *sr = b->sre + ((size_t)h * L + l) * dh, *si = b->sim + ((size_t)h * L + l) * dh;
                for (int d = 0; d < dh; d++) {
                    float r0 = sr[d], i0 = si[d];
                    float nr = cr * r0 - ci * i0 + kk * v[h * dh + d];
                    float ni = cr * i0 + ci * r0;
                    sr[d] = nr;
                    si[d] = ni;
                    rd[d] += qr * nr + qi * ni;
                }
            }
        }
        apply(m, &b->out, m->read, m->temporal);
        for (int g = 0; g < G; g++) {
            float nn = fmaxf(norm(z + g * W, W), EPS_NORMALIZE);
            for (int w = 0; w < W; w++) m->n[g * W + w] = z[g * W + w] / nn;
        }
        for (int g = 0; g < G; g++) {
            float gate = b->gate[g], cpl = b->coupling[g];
            for (int w = 0; w < W; w++) {
                float rot = 0;
                int pair = g / 2;
                if (pair < G / 2) {
                    float cm = b->cosm1[pair * W + w], sn = b->sinv[pair * W + w];
                    float first = m->n[2 * pair * W + w], second = m->n[(2 * pair + 1) * W + w];
                    rot = g % 2 == 0 ? cm * first - sn * second : sn * first + cm * second;
                }
                z[g * W + w] += gate * m->temporal[g * W + w] + (1 - gate) * (cpl * rot);
            }
        }
    }
    layer_norm(z, m->lnf_w, m->lnf_b, m->f, D);
    m->count++;
    for (int g = 0; g < G; g++) {
        float *s = m->sum + g * W, *fg = m->f + g * W, mean[W];
        for (int w = 0; w < W; w++) { s[w] += fg[w]; mean[w] = s[w] / m->count; }
        float nf = fmaxf(norm(fg, W), EPS_NORMALIZE), nm = fmaxf(norm(mean, W), EPS_NORMALIZE), c = 0;
        for (int w = 0; w < W; w++) c += (fg[w] / nf) * (mean[w] / nm);
        m->feat[D + g] = c;
        m->feat[D + G + g] = norm(mean, W);
    }
    memcpy(m->feat, m->f, D * 4);
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < 256; r++) {
        const float *w = m->head + (size_t)r * m->F;
        float s = 0;
        for (int j = 0; j < m->F; j++) s += w[j] * m->feat[j];
        m->logits[r] = s;
    }
    return m->logits;
}

static double bits(const float *logits, int target) {
    double mx = logits[0], s = 0;
    for (int i = 1; i < 256; i++) mx = fmax(mx, logits[i]);
    for (int i = 0; i < 256; i++) s += exp(logits[i] - mx);
    return (log(s) + mx - logits[target]) / log(2.0);
}

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s MODEL.bin check REF.bin | bench N | eval HELDOUT K\n", argv[0]); return 2; }
    Model m;
    load(&m, argv[1]);
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    printf("%s: %dL D=%d H=%d L=%d dh=%d, %s weights%s, %d threads\n", argv[1], m.layers, m.D, m.H, m.L, m.dh,
           m.ternary == 2 ? "int8" : m.ternary ? "ternary" : "float", m.act8 ? ", int8 projection inputs" : "", threads);
    if (!strcmp(argv[2], "check")) {
        FILE *f = fopen(argv[3], "rb");
        if (!f) { perror(argv[3]); return 1; }
        int32_t n;
        if (fread(&n, 4, 1, f) != 1) return 1;
        uint8_t *bytes = malloc(n);
        float *ref = malloc((size_t)n * 256 * 4);
        if (fread(bytes, 1, n, f) != (size_t)n || fread(ref, 4, (size_t)n * 256, f) != (size_t)n * 256) { fprintf(stderr, "short reference\n"); return 1; }
        fclose(f);
        double worst = 0, ours = 0, theirs = 0;
        int over = 0;
        for (int t = 0; t < n; t++) {
            const float *lg = step(&m, bytes[t]);
            double d = 0;
            for (int i = 0; i < 256; i++) d = fmax(d, fabs(lg[i] - ref[(size_t)t * 256 + i]));
            worst = fmax(worst, d);
            over += d > 1e-3;
            if (t + 1 < n) { ours += bits(lg, bytes[t + 1]); theirs += bits(ref + (size_t)t * 256, bytes[t + 1]); }
        }
        printf("%d bytes: max |logit diff| %.3e, positions over 1e-3: %d; bits per byte C %.6f, PyTorch %.6f, diff %+.2e\n",
               n, worst, over, ours / (n - 1), theirs / (n - 1), (ours - theirs) / (n - 1));
    } else if (!strcmp(argv[2], "eval")) {
        if (argc != 5) { fprintf(stderr, "usage: %s MODEL.bin eval HELDOUT K\n", argv[0]); return 2; }
        int k = atoi(argv[4]), win = 2048;
        FILE *f = fopen(argv[3], "rb");
        if (!f) { perror(argv[3]); return 1; }
        uint8_t *bytes = malloc((size_t)k * win);
        if (fread(bytes, 1, (size_t)k * win, f) != (size_t)k * win) { fprintf(stderr, "held-out file shorter than %d windows\n", k); return 1; }
        fclose(f);
        double total = 0, t0 = now();
        for (int w = 0; w < k; w++) {
            reset(&m);
            const uint8_t *b = bytes + (size_t)w * win;
            for (int t = 0; t + 1 < win; t++) total += bits(step(&m, b[t]), b[t + 1]);
        }
        printf("%d windows of %d bytes: %.6f bits per byte over %ld predictions, %.1f s\n", k, win, total / ((double)k * (win - 1)),
               (long)k * (win - 1), now() - t0);
    } else if (!strcmp(argv[2], "bench")) {
        int n = atoi(argv[3]);
        for (int t = 0; t < 64; t++) step(&m, (t * 37 + 11) & 255);   /* warm up */
        reset(&m);
        double t0 = now();
        for (int t = 0; t < n; t++) step(&m, (t * 37 + 11) & 255);
        double s = now() - t0;
        printf("%d bytes in %.3f s: %.1f bytes/s, %.3f ms a byte\n", n, s, n / s, 1e3 * s / n);
    } else {
        fprintf(stderr, "unknown mode %s\n", argv[2]);
        return 2;
    }
    return 0;
}
