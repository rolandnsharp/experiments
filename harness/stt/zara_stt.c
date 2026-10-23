/* zara_stt.c: streaming CPU inference for the oscillator-first recogniser (zara_stt.py).
 *
 *   cc -O3 -ffast-math -o zara_stt zara_stt.c -lm
 *   zara_stt MODEL.bin FILE.wav [--dump OUT.f32] [--bench] [--gain G]
 *
 * The model file comes from export_c.py. Audio is 16 kHz mono PCM16 (or float32) WAV; the whole
 * file is scaled to RMS 0.1 like the training batches (or by --gain G), zero-padded to whole
 * 20 ms frames, and run sample by sample through the oscillator bank and frame by frame through
 * the pole blocks: every state is a fixed-size array, nothing depends on the utterance length.
 * Prints the greedy CTC transcript; --dump writes the log-probabilities [frames][classes] for
 * comparison with export_c.py reference; --bench prints the real-time factor.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define GROUPS 7
#define NFFT 512
#define WINLEN 400
#define EPS_LN 1e-5f
#define EPS_NORM 1e-12f

typedef struct { int version, front, P, nin, D, H, L, dh, layers, classes, W, frame, rate, mlp; } Hdr;

typedef struct {
    float *ln_w, *ln_b, *qkv_w, *qkv_b, *qh_w, *kh_w, *spec, *logr, *theta, *out_w, *out_b, *cosm, *sinv, *coup, *gate;
    float *cr, *ci;        /* the pole per [H][L] */
    float *Sre, *Sim;      /* the state per [H][L][dh] */
    float *mln_w, *mln_b, *w1, *w3, *w2, *w2_b, mgate;   /* the gated MLP, if h.mlp > 0 */
} Block;

typedef struct {
    Hdr h;
    float *flogr, *ftheta, *fcr, *fci, *fgain, *zr, *zi, *energy;   /* the oscillator bank */
    float *mel, *win, *ring, *fr, *fi, *power;  int ring_len, ring_pos;  /* mel front: 512-point STFT, hop = frame */
    float *fln_w, *fln_b, *probe_w, *probe_b;
    Block *blk;
    float *lnf_w, *lnf_b, *head_w, *head_b;
    /* scratch */
    float *feat, *u, *z, *qkv, *qh, *kh, *read, *temporal, *n, *rot, *f, *logit, *h1, *h3;
    /* chunked path: K frames at a time, so every weight row is read once per chunk, not once per frame */
    int K; float *cFEAT, *cU, *cZ, *cQKV, *cREAD, *cTEMP, *cF, *cLOG, *cH1, *cH3;
} Model;

static float *arr(FILE *f, size_t n) {
    float *p = malloc(n * sizeof(float));
    if (!p || fread(p, sizeof(float), n, f) != n) { fprintf(stderr, "short model file\n"); exit(1); }
    return p;
}

static Model *load_model(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "ZSTT", 4)) { fprintf(stderr, "not a ZSTT model\n"); exit(1); }
    Model *m = calloc(1, sizeof(Model));
    if (fread(&m->h, sizeof(int32_t), 13, f) != 13) { fprintf(stderr, "short header\n"); exit(1); }
    m->h.mlp = 0;
    if (m->h.version >= 2 && fread(&m->h.mlp, sizeof(int32_t), 1, f) != 1) { fprintf(stderr, "short header\n"); exit(1); }
    Hdr h = m->h;
    int D = h.D, H = h.H, L = h.L, dh = h.dh, W = h.W, pairs = GROUPS / 2;
    if (h.front == 0) {
        m->flogr = arr(f, h.P); m->ftheta = arr(f, h.P);
        m->fcr = malloc(h.P * 4); m->fci = malloc(h.P * 4); m->fgain = malloc(h.P * 4);
        for (int p = 0; p < h.P; p++) {
            float r = expf(m->flogr[p]);
            m->fcr[p] = r * cosf(m->ftheta[p]); m->fci[p] = r * sinf(m->ftheta[p]); m->fgain[p] = 1 - r;
        }
        m->zr = calloc(h.P, 4); m->zi = calloc(h.P, 4); m->energy = calloc(h.P, 4);
    } else {
        m->mel = arr(f, (size_t)h.P * (NFFT / 2 + 1));
        m->win = calloc(NFFT, 4);
        for (int n = 0; n < WINLEN; n++) m->win[(NFFT - WINLEN) / 2 + n] = 0.5f - 0.5f * cosf(2 * (float)M_PI * n / WINLEN);   /* periodic Hann, centred in the FFT frame */
        m->ring_len = 2 * NFFT; m->ring = calloc(m->ring_len, 4); m->ring_pos = 0;
        m->fr = malloc(NFFT * 4); m->fi = malloc(NFFT * 4); m->power = malloc((NFFT / 2 + 1) * 4);
    }
    m->fln_w = arr(f, h.nin); m->fln_b = arr(f, h.nin); m->probe_w = arr(f, (size_t)D * h.nin); m->probe_b = arr(f, D);
    m->blk = calloc(h.layers, sizeof(Block));
    for (int i = 0; i < h.layers; i++) {
        Block *b = &m->blk[i];
        b->ln_w = arr(f, D); b->ln_b = arr(f, D); b->qkv_w = arr(f, (size_t)3 * D * D); b->qkv_b = arr(f, 3 * D);
        b->qh_w = arr(f, (size_t)H * 2 * L * dh); b->kh_w = arr(f, (size_t)H * L * dh); b->spec = arr(f, H * L);
        b->logr = arr(f, H * L); b->theta = arr(f, H * L);
        b->out_w = arr(f, (size_t)D * D); b->out_b = arr(f, D);
        b->cosm = arr(f, pairs * W); b->sinv = arr(f, pairs * W); b->coup = arr(f, GROUPS); b->gate = arr(f, GROUPS);
        b->cr = malloc(H * L * 4); b->ci = malloc(H * L * 4);
        for (int j = 0; j < H * L; j++) { float r = expf(b->logr[j]); b->cr[j] = r * cosf(b->theta[j]); b->ci[j] = r * sinf(b->theta[j]); }
        b->Sre = calloc((size_t)H * L * dh, 4); b->Sim = calloc((size_t)H * L * dh, 4);
        if (h.mlp) {
            b->mln_w = arr(f, D); b->mln_b = arr(f, D); b->w1 = arr(f, (size_t)h.mlp * D); b->w3 = arr(f, (size_t)h.mlp * D);
            b->w2 = arr(f, (size_t)D * h.mlp); b->w2_b = arr(f, D);
            float *g = arr(f, 1); b->mgate = g[0]; free(g);
        }
    }
    m->lnf_w = arr(f, D); m->lnf_b = arr(f, D); m->head_w = arr(f, (size_t)h.classes * D); m->head_b = arr(f, h.classes);
    fclose(f);
    m->feat = calloc(h.nin, 4); m->u = malloc(D * 4); m->z = malloc(D * 4); m->qkv = malloc(3 * D * 4);
    m->qh = malloc(H * 2 * L * 4); m->kh = malloc(H * L * 4); m->read = malloc(D * 4); m->temporal = malloc(D * 4);
    m->n = malloc(D * 4); m->rot = calloc(D, 4); m->f = malloc(D * 4); m->logit = malloc(h.classes * 4);
    if (h.mlp) { m->h1 = malloc(h.mlp * 4); m->h3 = malloc(h.mlp * 4); }
    return m;
}

static void layer_norm(const float *x, const float *w, const float *b, float *y, int n) {
    float mean = 0, var = 0;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;
    for (int i = 0; i < n; i++) { float d = x[i] - mean; var += d * d; }
    var /= n;
    float s = 1.0f / sqrtf(var + EPS_LN);
    for (int i = 0; i < n; i++) y[i] = (x[i] - mean) * s * w[i] + b[i];
}

/* y[o] = b[o] + sum_i W[o][i] x[i] */
static void matvec(const float *W, const float *x, const float *b, float *y, int out, int in) {
    for (int o = 0; o < out; o++) {
        const float *w = W + (size_t)o * in;
        float s = 0;
        for (int i = 0; i < in; i++) s += w[i] * x[i];
        y[o] = s + (b ? b[o] : 0);
    }
}

static void block_step(Model *m, Block *b, float *z) {
    Hdr h = m->h;
    int D = h.D, H = h.H, L = h.L, dh = h.dh, W = h.W, pairs = GROUPS / 2;
    layer_norm(z, b->ln_w, b->ln_b, m->u, D);
    matvec(b->qkv_w, m->u, b->qkv_b, m->qkv, 3 * D, D);
    const float *q = m->qkv, *k = m->qkv + D, *v = m->qkv + 2 * D;
    for (int hh = 0; hh < H; hh++) {
        const float *qd = q + hh * dh, *kd = k + hh * dh, *vd = v + hh * dh;
        for (int l = 0; l < 2 * L; l++) {
            const float *w = b->qh_w + ((size_t)hh * 2 * L + l) * dh;
            float s = 0;
            for (int d = 0; d < dh; d++) s += w[d] * qd[d];
            m->qh[hh * 2 * L + l] = s;
        }
        for (int l = 0; l < L; l++) {
            const float *w = b->kh_w + ((size_t)hh * L + l) * dh;
            float s = 0;
            for (int d = 0; d < dh; d++) s += w[d] * kd[d];
            m->kh[hh * L + l] = s * b->spec[hh * L + l];
        }
        float *rd = m->read + hh * dh;
        for (int d = 0; d < dh; d++) rd[d] = 0;
        for (int l = 0; l < L; l++) {
            int j = hh * L + l;
            float cr = b->cr[j], ci = b->ci[j], kh = m->kh[j];
            float qre = m->qh[hh * 2 * L + l], qim = m->qh[hh * 2 * L + L + l];
            float *Sre = b->Sre + (size_t)j * dh, *Sim = b->Sim + (size_t)j * dh;
            for (int d = 0; d < dh; d++) {
                float re = cr * Sre[d] - ci * Sim[d] + kh * vd[d];
                float im = cr * Sim[d] + ci * Sre[d];
                Sre[d] = re; Sim[d] = im;
                rd[d] += qre * re + qim * im;
            }
        }
    }
    matvec(b->out_w, m->read, b->out_b, m->temporal, D, D);
    /* spatial: unit vectors per group, pairs rotated, the last group zero */
    for (int g = 0; g < GROUPS; g++) {
        const float *zg = z + g * W;
        float s = 0;
        for (int w = 0; w < W; w++) s += zg[w] * zg[w];
        s = sqrtf(s); if (s < EPS_NORM) s = EPS_NORM;
        for (int w = 0; w < W; w++) m->n[g * W + w] = zg[w] / s;
    }
    for (int i = 0; i < pairs; i++) {
        const float *first = m->n + (2 * i) * W, *second = m->n + (2 * i + 1) * W;
        const float *cm = b->cosm + i * W, *sn = b->sinv + i * W;
        float *r0 = m->rot + (2 * i) * W, *r1 = m->rot + (2 * i + 1) * W;
        for (int w = 0; w < W; w++) {
            r0[w] = cm[w] * first[w] - sn[w] * second[w];
            r1[w] = sn[w] * first[w] + cm[w] * second[w];
        }
    }
    for (int g = 0; g < GROUPS; g++) {
        float gate = b->gate[g], coup = b->coup[g];
        for (int w = 0; w < W; w++) {
            int i = g * W + w;
            float spatial = g < 2 * pairs ? coup * m->rot[i] : 0;
            z[i] += gate * m->temporal[i] + (1 - gate) * spatial;
        }
    }
}

/* z += gate * W2 (silu(W1 u) * W3 u), u = ln(z) */
static void mlp_step(Model *m, Block *b, float *z) {
    int D = m->h.D, Hm = m->h.mlp;
    layer_norm(z, b->mln_w, b->mln_b, m->u, D);
    matvec(b->w1, m->u, NULL, m->h1, Hm, D);
    matvec(b->w3, m->u, NULL, m->h3, Hm, D);
    for (int j = 0; j < Hm; j++) { float a = m->h1[j]; m->h1[j] = a / (1 + expf(-a)) * m->h3[j]; }
    matvec(b->w2, m->h1, b->w2_b, m->temporal, D, Hm);
    for (int i = 0; i < D; i++) z[i] += b->mgate * m->temporal[i];
}

/* one frame of features -> log-probabilities over the classes */
static void frame_step(Model *m, const float *feat, float *logp) {
    Hdr h = m->h;
    layer_norm(feat, m->fln_w, m->fln_b, m->feat, h.nin);
    matvec(m->probe_w, m->feat, m->probe_b, m->z, h.D, h.nin);
    for (int i = 0; i < h.layers; i++) {
        block_step(m, &m->blk[i], m->z);
        if (h.mlp) mlp_step(m, &m->blk[i], m->z);
    }
    layer_norm(m->z, m->lnf_w, m->lnf_b, m->f, h.D);
    matvec(m->head_w, m->f, m->head_b, m->logit, h.classes, h.D);
    float mx = m->logit[0];
    for (int c = 1; c < h.classes; c++) if (m->logit[c] > mx) mx = m->logit[c];
    float s = 0;
    for (int c = 0; c < h.classes; c++) s += expf(m->logit[c] - mx);
    float lse = mx + logf(s);
    for (int c = 0; c < h.classes; c++) logp[c] = m->logit[c] - lse;
}

/* the oscillator bank: one sample in; at a frame end, the 3P features out */
static void bank_sample(Model *m, float x) {
    int P = m->h.P;
    for (int p = 0; p < P; p++) {
        float re = m->fcr[p] * m->zr[p] - m->fci[p] * m->zi[p] + x;
        float im = m->fcr[p] * m->zi[p] + m->fci[p] * m->zr[p];
        m->zr[p] = re; m->zi[p] = im;
        float gr = m->fgain[p] * re, gi = m->fgain[p] * im;
        m->energy[p] += gr * gr + gi * gi;
    }
}

static void bank_frame(Model *m, float *feat) {
    int P = m->h.P;
    for (int p = 0; p < P; p++) {
        feat[p] = m->fgain[p] * m->zr[p];
        feat[P + p] = m->fgain[p] * m->zi[p];
        feat[2 * P + p] = logf(m->energy[p] / m->h.frame + 1e-8f);
        m->energy[p] = 0;
    }
}

/* in-place radix-2 complex FFT, n a power of two */
static void fft(float *re, float *im, int n) {
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { float t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
    }
    for (int len = 2; len <= n; len <<= 1) {
        float ang = -2 * (float)M_PI / len, wr = cosf(ang), wi = sinf(ang);
        for (int i = 0; i < n; i += len) {
            float cr = 1, ci = 0;
            for (int j = 0; j < len / 2; j++) {
                float ur = re[i + j], ui = im[i + j];
                float vr = re[i + j + len / 2] * cr - im[i + j + len / 2] * ci;
                float vi = re[i + j + len / 2] * ci + im[i + j + len / 2] * cr;
                re[i + j] = ur + vr; im[i + j] = ui + vi; re[i + j + len / 2] = ur - vr; im[i + j + len / 2] = ui - vi;
                float t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t;
            }
        }
    }
}

/* mel front: push one sample; `total` counts samples pushed so far (for reflect padding at the start) */
static void mel_sample(Model *m, float x) {
    m->ring[m->ring_pos] = x;
    m->ring_pos = (m->ring_pos + 1) % m->ring_len;
}

/* sample index s (may be negative: reflected) -> value, given `total` samples pushed, the newest at ring_pos-1 */
static float mel_at(Model *m, long s, long total) {
    if (s < 0) s = -s;                           /* torch.stft center=True pads by reflection */
    if (s >= total) return 0;                    /* only reached by frames past the end, which are not scored */
    long back = total - 1 - s;                   /* how far behind the newest sample */
    if (back >= m->ring_len) return 0;
    return m->ring[((m->ring_pos - 1 - back) % m->ring_len + m->ring_len) % m->ring_len];
}

/* features for frame t: log mel energies of the window centred on sample t*frame */
static void mel_frame(Model *m, long t, long total, float *feat) {
    long centre = t * m->h.frame;
    for (int n = 0; n < NFFT; n++) { m->fr[n] = m->win[n] * mel_at(m, centre - NFFT / 2 + n, total); m->fi[n] = 0; }
    fft(m->fr, m->fi, NFFT);
    for (int k = 0; k <= NFFT / 2; k++) m->power[k] = m->fr[k] * m->fr[k] + m->fi[k] * m->fi[k];
    for (int j = 0; j < m->h.P; j++) {
        const float *row = m->mel + (size_t)j * (NFFT / 2 + 1);
        float e = 0;
        for (int k = 0; k <= NFFT / 2; k++) e += row[k] * m->power[k];
        feat[j] = logf(e + 1e-8f);
    }
}

static float *read_wav(const char *path, size_t *n_out, int rate) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END); long size = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char *d = malloc(size);
    if (fread(d, 1, size, f) != (size_t)size) { fprintf(stderr, "short wav\n"); exit(1); }
    fclose(f);
    if (size < 12 || memcmp(d, "RIFF", 4) || memcmp(d + 8, "WAVE", 4)) { fprintf(stderr, "%s: not a WAV\n", path); exit(1); }
    int fmt = 0, ch = 0, sr = 0, bits = 0; unsigned char *pcm = NULL; size_t pcm_size = 0;
    for (long pos = 12; pos + 8 <= size;) {
        uint32_t csz; memcpy(&csz, d + pos + 4, 4);
        if (!memcmp(d + pos, "fmt ", 4)) {
            uint16_t t, c, b; uint32_t r;
            memcpy(&t, d + pos + 8, 2); memcpy(&c, d + pos + 10, 2); memcpy(&r, d + pos + 12, 4); memcpy(&b, d + pos + 22, 2);
            fmt = t; ch = c; sr = r; bits = b;
        } else if (!memcmp(d + pos, "data", 4)) { pcm = d + pos + 8; pcm_size = csz; if (pos + 8 + (long)csz > size) pcm_size = size - pos - 8; }
        pos += 8 + csz + (csz & 1);
    }
    if (!pcm || sr != rate || ch != 1) { fprintf(stderr, "%s: need %d Hz mono PCM16/float32 WAV (got %d Hz, %d ch, fmt %d, %d bits)\n", path, rate, sr, ch, fmt, bits); exit(1); }
    size_t n; float *x;
    if (fmt == 1 && bits == 16) {
        n = pcm_size / 2; x = malloc(n * 4);
        for (size_t i = 0; i < n; i++) { int16_t s; memcpy(&s, pcm + 2 * i, 2); x[i] = s / 32768.0f; }
    } else if (fmt == 3 && bits == 32) {
        n = pcm_size / 4; x = malloc(n * 4); memcpy(x, pcm, n * 4);
    } else { fprintf(stderr, "%s: unsupported sample format\n", path); exit(1); }
    free(d);
    *n_out = n;
    return x;
}

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

/* Y[k][o] = b[o] + sum_i W[o][i] X[k][i] for k < K: each weight row is loaded once and used K times */
static float dot(const float *w, const float *x, int in) {
    float s = 0;
    for (int i = 0; i < in; i++) s += w[i] * x[i];
    return s;
}

#if defined(__aarch64__)
#include <arm_neon.h>
/* 4 output rows x 4 frames per pass: 8 vector loads feed 16 fused multiply-adds */
static void matmul_f32(const float *W, const float *X, const float *b, float *Y, int out, int in, int K) {
    int o = 0;
    if (in % 4 == 0) {
        for (; o + 4 <= out; o += 4) {
            const float *w0 = W + (size_t)o * in, *w1 = w0 + in, *w2 = w1 + in, *w3 = w2 + in;
            int k = 0;
            for (; k + 4 <= K; k += 4) {
                const float *x0 = X + (size_t)k * in, *x1 = x0 + in, *x2 = x1 + in, *x3 = x2 + in;
                float32x4_t a00 = vdupq_n_f32(0), a01 = a00, a02 = a00, a03 = a00, a10 = a00, a11 = a00, a12 = a00, a13 = a00;
                float32x4_t a20 = a00, a21 = a00, a22 = a00, a23 = a00, a30 = a00, a31 = a00, a32 = a00, a33 = a00;
                for (int i = 0; i < in; i += 4) {
                    float32x4_t v0 = vld1q_f32(x0 + i), v1 = vld1q_f32(x1 + i), v2 = vld1q_f32(x2 + i), v3 = vld1q_f32(x3 + i);
                    float32x4_t u = vld1q_f32(w0 + i);
                    a00 = vfmaq_f32(a00, u, v0); a01 = vfmaq_f32(a01, u, v1); a02 = vfmaq_f32(a02, u, v2); a03 = vfmaq_f32(a03, u, v3);
                    u = vld1q_f32(w1 + i);
                    a10 = vfmaq_f32(a10, u, v0); a11 = vfmaq_f32(a11, u, v1); a12 = vfmaq_f32(a12, u, v2); a13 = vfmaq_f32(a13, u, v3);
                    u = vld1q_f32(w2 + i);
                    a20 = vfmaq_f32(a20, u, v0); a21 = vfmaq_f32(a21, u, v1); a22 = vfmaq_f32(a22, u, v2); a23 = vfmaq_f32(a23, u, v3);
                    u = vld1q_f32(w3 + i);
                    a30 = vfmaq_f32(a30, u, v0); a31 = vfmaq_f32(a31, u, v1); a32 = vfmaq_f32(a32, u, v2); a33 = vfmaq_f32(a33, u, v3);
                }
                float b0 = b ? b[o] : 0, b1 = b ? b[o + 1] : 0, b2 = b ? b[o + 2] : 0, b3 = b ? b[o + 3] : 0;
                float *y0 = Y + (size_t)k * out + o, *y1 = y0 + out, *y2 = y1 + out, *y3 = y2 + out;
                y0[0] = vaddvq_f32(a00) + b0; y1[0] = vaddvq_f32(a01) + b0; y2[0] = vaddvq_f32(a02) + b0; y3[0] = vaddvq_f32(a03) + b0;
                y0[1] = vaddvq_f32(a10) + b1; y1[1] = vaddvq_f32(a11) + b1; y2[1] = vaddvq_f32(a12) + b1; y3[1] = vaddvq_f32(a13) + b1;
                y0[2] = vaddvq_f32(a20) + b2; y1[2] = vaddvq_f32(a21) + b2; y2[2] = vaddvq_f32(a22) + b2; y3[2] = vaddvq_f32(a23) + b2;
                y0[3] = vaddvq_f32(a30) + b3; y1[3] = vaddvq_f32(a31) + b3; y2[3] = vaddvq_f32(a32) + b3; y3[3] = vaddvq_f32(a33) + b3;
            }
            for (; k < K; k++) {
                const float *x = X + (size_t)k * in;
                for (int r = 0; r < 4; r++) Y[(size_t)k * out + o + r] = dot(W + (size_t)(o + r) * in, x, in) + (b ? b[o + r] : 0);
            }
        }
    }
    for (; o < out; o++)
        for (int k = 0; k < K; k++) Y[(size_t)k * out + o] = dot(W + (size_t)o * in, X + (size_t)k * in, in) + (b ? b[o] : 0);
}
#else
static void matmul_f32(const float *W, const float *X, const float *b, float *Y, int out, int in, int K) {
    for (int o = 0; o < out; o++)
        for (int k = 0; k < K; k++) Y[(size_t)k * out + o] = dot(W + (size_t)o * in, X + (size_t)k * in, in) + (b ? b[o] : 0);
}
#endif

/* --int8: weights quantised per output row (symmetric, scale = max|w| / 127) at load time,
 * activations per frame row; products accumulate exactly in int32 and are rescaled once.
 * The A72 has no int8 dot-product instruction, so the kernel uses widening multiplies
 * (vmull_s8 / vmlal_high_s8 into int16, pairwise-accumulated into int32). */
static int g_int8 = 0;
typedef struct { const float *f; int8_t *q; float *scale; int out, in; } QMat;
#define MAXQ 64
static QMat g_q[MAXQ]; static int g_nq = 0;
static int8_t *g_xq; static float *g_xs; static size_t g_xqcap = 0;

static QMat *qmat_of(const float *W, int out, int in) {
    for (int i = 0; i < g_nq; i++) if (g_q[i].f == W) return &g_q[i];
    if (g_nq == MAXQ || in % 16) return NULL;
    QMat *q = &g_q[g_nq++]; q->f = W; q->out = out; q->in = in;
    q->q = malloc((size_t)out * in); q->scale = malloc(out * 4);
    for (int o = 0; o < out; o++) {
        const float *w = W + (size_t)o * in; float mx = 0;
        for (int i = 0; i < in; i++) mx = fabsf(w[i]) > mx ? fabsf(w[i]) : mx;
        float sc = mx > 0 ? mx / 127.0f : 1.0f; q->scale[o] = sc;
        for (int i = 0; i < in; i++) { int v = (int)lrintf(w[i] / sc); q->q[(size_t)o * in + i] = (int8_t)(v > 127 ? 127 : v < -127 ? -127 : v); }
    }
    return q;
}

static void quantise_rows(const float *X, int K, int in) {
    if ((size_t)K * in > g_xqcap) { g_xqcap = (size_t)K * in; g_xq = realloc(g_xq, g_xqcap); g_xs = realloc(g_xs, K * 4); }
    for (int k = 0; k < K; k++) {
        const float *x = X + (size_t)k * in; float mx = 0;
        for (int i = 0; i < in; i++) mx = fabsf(x[i]) > mx ? fabsf(x[i]) : mx;
        float sc = mx > 0 ? mx / 127.0f : 1.0f, inv = 1.0f / sc; g_xs[k] = sc;
        for (int i = 0; i < in; i++) { int v = (int)lrintf(x[i] * inv); g_xq[(size_t)k * in + i] = (int8_t)(v > 127 ? 127 : v < -127 ? -127 : v); }
    }
}

static int32_t idot(const int8_t *a, const int8_t *b, int n) { int32_t s = 0; for (int i = 0; i < n; i++) s += a[i] * b[i]; return s; }

static void matmul_q8(const QMat *q, const float *X, const float *b, float *Y, int K) {
    int out = q->out, in = q->in;
    quantise_rows(X, K, in);
    int o = 0;
#if defined(__aarch64__)
    for (; o + 4 <= out; o += 4) {
        const int8_t *w0 = q->q + (size_t)o * in, *w1 = w0 + in, *w2 = w1 + in, *w3 = w2 + in;
        int k = 0;
        for (; k + 4 <= K; k += 4) {
            const int8_t *x0 = g_xq + (size_t)k * in, *x1 = x0 + in, *x2 = x1 + in, *x3 = x2 + in;
            int32x4_t acc[4][4];
            for (int r = 0; r < 4; r++) for (int j = 0; j < 4; j++) acc[r][j] = vdupq_n_s32(0);
            for (int i = 0; i < in; i += 16) {
                int8x16_t v[4] = { vld1q_s8(x0 + i), vld1q_s8(x1 + i), vld1q_s8(x2 + i), vld1q_s8(x3 + i) };
                const int8_t *wr[4] = { w0 + i, w1 + i, w2 + i, w3 + i };
                for (int r = 0; r < 4; r++) {
                    int8x16_t u = vld1q_s8(wr[r]);
                    for (int j = 0; j < 4; j++) {
                        int16x8_t pr = vmull_s8(vget_low_s8(u), vget_low_s8(v[j]));
                        pr = vmlal_high_s8(pr, u, v[j]);
                        acc[r][j] = vpadalq_s16(acc[r][j], pr);
                    }
                }
            }
            for (int r = 0; r < 4; r++) for (int j = 0; j < 4; j++)
                Y[(size_t)(k + j) * out + o + r] = (float)vaddvq_s32(acc[r][j]) * q->scale[o + r] * g_xs[k + j] + (b ? b[o + r] : 0);
        }
        for (; k < K; k++) for (int r = 0; r < 4; r++)
            Y[(size_t)k * out + o + r] = (float)idot(q->q + (size_t)(o + r) * in, g_xq + (size_t)k * in, in) * q->scale[o + r] * g_xs[k] + (b ? b[o + r] : 0);
    }
#endif
    for (; o < out; o++) for (int k = 0; k < K; k++)
        Y[(size_t)k * out + o] = (float)idot(q->q + (size_t)o * in, g_xq + (size_t)k * in, in) * q->scale[o] * g_xs[k] + (b ? b[o] : 0);
}

static void matmul(const float *W, const float *X, const float *b, float *Y, int out, int in, int K) {
    QMat *q = g_int8 ? qmat_of(W, out, in) : NULL;
    if (q) matmul_q8(q, X, b, Y, K); else matmul_f32(W, X, b, Y, out, in, K);
}

static void chunk_alloc(Model *m, int K) {
    Hdr h = m->h; int D = h.D;
    m->K = K;
    m->cFEAT = malloc((size_t)K * h.nin * 4); m->cU = malloc((size_t)K * D * 4); m->cZ = malloc((size_t)K * D * 4);
    m->cQKV = malloc((size_t)K * 3 * D * 4); m->cREAD = malloc((size_t)K * D * 4); m->cTEMP = malloc((size_t)K * D * 4);
    m->cF = malloc((size_t)K * D * 4); m->cLOG = malloc((size_t)K * h.classes * 4);
    if (h.mlp) { m->cH1 = malloc((size_t)K * h.mlp * 4); m->cH3 = malloc((size_t)K * h.mlp * 4); }
}

static void block_chunk(Model *m, Block *b, int K) {
    Hdr h = m->h;
    int D = h.D, H = h.H, L = h.L, dh = h.dh, W = h.W, pairs = GROUPS / 2;
    for (int k = 0; k < K; k++) layer_norm(m->cZ + (size_t)k * D, b->ln_w, b->ln_b, m->cU + (size_t)k * D, D);
    matmul(b->qkv_w, m->cU, b->qkv_b, m->cQKV, 3 * D, D, K);
    for (int k = 0; k < K; k++) {          /* the recurrence is sequential in time */
        const float *q = m->cQKV + (size_t)k * 3 * D, *kk = q + D, *v = q + 2 * D;
        float *read = m->cREAD + (size_t)k * D;
        for (int hh = 0; hh < H; hh++) {
            const float *qd = q + hh * dh, *kd = kk + hh * dh, *vd = v + hh * dh;
            for (int l = 0; l < 2 * L; l++) {
                const float *w = b->qh_w + ((size_t)hh * 2 * L + l) * dh;
                float s2 = 0;
                for (int d = 0; d < dh; d++) s2 += w[d] * qd[d];
                m->qh[hh * 2 * L + l] = s2;
            }
            for (int l = 0; l < L; l++) {
                const float *w = b->kh_w + ((size_t)hh * L + l) * dh;
                float s2 = 0;
                for (int d = 0; d < dh; d++) s2 += w[d] * kd[d];
                m->kh[hh * L + l] = s2 * b->spec[hh * L + l];
            }
            float *rd = read + hh * dh;
            for (int d = 0; d < dh; d++) rd[d] = 0;
            for (int l = 0; l < L; l++) {
                int j = hh * L + l;
                float cr = b->cr[j], ci = b->ci[j], kh = m->kh[j];
                float qre = m->qh[hh * 2 * L + l], qim = m->qh[hh * 2 * L + L + l];
                float *Sre = b->Sre + (size_t)j * dh, *Sim = b->Sim + (size_t)j * dh;
                for (int d = 0; d < dh; d++) {
                    float re = cr * Sre[d] - ci * Sim[d] + kh * vd[d];
                    float im = cr * Sim[d] + ci * Sre[d];
                    Sre[d] = re; Sim[d] = im;
                    rd[d] += qre * re + qim * im;
                }
            }
        }
    }
    matmul(b->out_w, m->cREAD, b->out_b, m->cTEMP, D, D, K);
    for (int k = 0; k < K; k++) {
        float *z = m->cZ + (size_t)k * D, *temporal = m->cTEMP + (size_t)k * D;
        for (int g = 0; g < GROUPS; g++) {
            const float *zg = z + g * W;
            float s2 = 0;
            for (int w = 0; w < W; w++) s2 += zg[w] * zg[w];
            s2 = sqrtf(s2); if (s2 < EPS_NORM) s2 = EPS_NORM;
            for (int w = 0; w < W; w++) m->n[g * W + w] = zg[w] / s2;
        }
        for (int i = 0; i < pairs; i++) {
            const float *first = m->n + (2 * i) * W, *second = m->n + (2 * i + 1) * W;
            const float *cm = b->cosm + i * W, *sn = b->sinv + i * W;
            float *r0 = m->rot + (2 * i) * W, *r1 = m->rot + (2 * i + 1) * W;
            for (int w = 0; w < W; w++) {
                r0[w] = cm[w] * first[w] - sn[w] * second[w];
                r1[w] = sn[w] * first[w] + cm[w] * second[w];
            }
        }
        for (int g = 0; g < GROUPS; g++) {
            float gate = b->gate[g], coup = b->coup[g];
            for (int w = 0; w < W; w++) {
                int i = g * W + w;
                float spatial = g < 2 * pairs ? coup * m->rot[i] : 0;
                z[i] += gate * temporal[i] + (1 - gate) * spatial;
            }
        }
    }
}

static void mlp_chunk(Model *m, Block *b, int K) {
    int D = m->h.D, Hm = m->h.mlp;
    for (int k = 0; k < K; k++) layer_norm(m->cZ + (size_t)k * D, b->mln_w, b->mln_b, m->cU + (size_t)k * D, D);
    matmul(b->w1, m->cU, NULL, m->cH1, Hm, D, K);
    matmul(b->w3, m->cU, NULL, m->cH3, Hm, D, K);
    for (size_t j = 0; j < (size_t)K * Hm; j++) { float a = m->cH1[j]; m->cH1[j] = a / (1 + expf(-a)) * m->cH3[j]; }
    matmul(b->w2, m->cH1, b->w2_b, m->cTEMP, D, Hm, K);
    for (size_t i = 0; i < (size_t)K * D; i++) m->cZ[i] += b->mgate * m->cTEMP[i];
}

/* K frames of features (m->cFEAT) -> K rows of log-probabilities */
static void frames_chunk(Model *m, int K, float *logp) {
    Hdr h = m->h; int D = h.D, C = h.classes;
    for (int k = 0; k < K; k++) layer_norm(m->cFEAT + (size_t)k * h.nin, m->fln_w, m->fln_b, m->cFEAT + (size_t)k * h.nin, h.nin);
    matmul(m->probe_w, m->cFEAT, m->probe_b, m->cZ, D, h.nin, K);
    for (int i = 0; i < h.layers; i++) {
        block_chunk(m, &m->blk[i], K);
        if (h.mlp) mlp_chunk(m, &m->blk[i], K);
    }
    for (int k = 0; k < K; k++) layer_norm(m->cZ + (size_t)k * D, m->lnf_w, m->lnf_b, m->cF + (size_t)k * D, D);
    matmul(m->head_w, m->cF, m->head_b, m->cLOG, C, D, K);
    for (int k = 0; k < K; k++) {
        const float *lg = m->cLOG + (size_t)k * C; float *lp = logp + (size_t)k * C;
        float mx = lg[0];
        for (int c = 1; c < C; c++) if (lg[c] > mx) mx = lg[c];
        float s2 = 0;
        for (int c = 0; c < C; c++) s2 += expf(lg[c] - mx);
        float lse = mx + logf(s2);
        for (int c = 0; c < C; c++) lp[c] = lg[c] - lse;
    }
}


/* ---------------- CTC prefix beam search with a word 3-gram (--lm) ----------------
 * Prefixes are nodes of a tree shared by all beams (node = parent + one character), so two
 * beams with the same text are the same node and merge exactly. Each node caches its word-LM
 * state, which depends only on its text: the last two completed words and the accumulated
 * alpha * log p(word | context) + beta per word, plus a one-off penalty when the partial word
 * is not a prefix of any vocabulary word. Log-probabilities are natural logs throughout. */
typedef struct { int V, N2, N3; char **w; int *wl; float *ulp, *ubo; uint64_t *k2; float *l2, *b2; uint64_t *k3; float *l3; int unk, bos, eos; } LM;

static int lm_cmp(const LM *lm, int i, const char *s, int n) {   /* sign of (word i) - (s, n), byte order */
    int m = lm->wl[i] < n ? lm->wl[i] : n, c = memcmp(lm->w[i], s, m);
    return c ? c : lm->wl[i] - n;
}
static int lm_lower(const LM *lm, const char *s, int n) {
    int lo = 0, hi = lm->V;
    while (lo < hi) { int mid = (lo + hi) / 2; if (lm_cmp(lm, mid, s, n) < 0) lo = mid + 1; else hi = mid; }
    return lo;
}
static int lm_id(const LM *lm, const char *s, int n) {
    int i = lm_lower(lm, s, n);
    return i < lm->V && lm_cmp(lm, i, s, n) == 0 ? i : -1;
}
static int lm_is_prefix(const LM *lm, const char *s, int n) {
    int i = lm_lower(lm, s, n);
    return i < lm->V && lm->wl[i] >= n && memcmp(lm->w[i], s, n) == 0;
}
static long find_key(const uint64_t *k, long n, uint64_t key) {
    long lo = 0, hi = n;
    while (lo < hi) { long mid = (lo + hi) / 2; if (k[mid] < key) lo = mid + 1; else hi = mid; }
    return lo < n && k[lo] == key ? lo : -1;
}
/* log p(w3 | w1 w2) with Katz backoff; w1 or w2 may be -1 (no context) */
static float lm_score(const LM *lm, int w1, int w2, int w3) {
    if (w1 >= 0 && w2 >= 0) {
        long i = find_key(lm->k3, lm->N3, ((uint64_t)w1 << 36) | ((uint64_t)w2 << 18) | (uint64_t)w3);
        if (i >= 0) return lm->l3[i];
        long j = find_key(lm->k2, lm->N2, ((uint64_t)w1 << 18) | (uint64_t)w2);
        float bo = j >= 0 ? lm->b2[j] : 0;
        return bo + lm_score(lm, -1, w2, w3);
    }
    if (w2 >= 0) {
        long i = find_key(lm->k2, lm->N2, ((uint64_t)w2 << 18) | (uint64_t)w3);
        if (i >= 0) return lm->l2[i];
        return lm->ubo[w2] + lm->ulp[w3];
    }
    return lm->ulp[w3];
}
static LM *load_lm(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    char magic[4]; int hdr[3];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "ZLM1", 4) || fread(hdr, 4, 3, f) != 3) { fprintf(stderr, "%s: not a ZLM1 LM\n", path); exit(1); }
    LM *lm = calloc(1, sizeof(LM)); lm->V = hdr[0]; lm->N2 = hdr[1]; lm->N3 = hdr[2];
    lm->w = malloc(lm->V * sizeof(char *)); lm->wl = malloc(lm->V * sizeof(int));
    for (int i = 0; i < lm->V; i++) {
        uint16_t n; if (fread(&n, 2, 1, f) != 1) { fprintf(stderr, "short LM\n"); exit(1); }
        lm->w[i] = malloc(n + 1); lm->wl[i] = n;
        if (fread(lm->w[i], 1, n, f) != n) { fprintf(stderr, "short LM\n"); exit(1); }
        lm->w[i][n] = 0;
    }
#define RD(ptr, type, cnt) do { ptr = malloc((size_t)(cnt) * sizeof(type)); if (fread(ptr, sizeof(type), (cnt), f) != (size_t)(cnt)) { fprintf(stderr, "short LM\n"); exit(1); } } while (0)
    RD(lm->ulp, float, lm->V); RD(lm->ubo, float, lm->V);
    RD(lm->k2, uint64_t, lm->N2); RD(lm->l2, float, lm->N2); RD(lm->b2, float, lm->N2);
    RD(lm->k3, uint64_t, lm->N3); RD(lm->l3, float, lm->N3);
#undef RD
    fclose(f);
    lm->unk = lm_id(lm, "<unk>", 5); if (lm->unk < 0) lm->unk = lm_id(lm, "<UNK>", 5);
    if (lm->unk < 0) { fprintf(stderr, "%s: no <unk> word\n", path); exit(1); }
    lm->bos = lm_id(lm, "<s>", 3); lm->eos = lm_id(lm, "</s>", 4);
    return lm;
}

typedef struct { int parent, depth, wstart, w1, w2, penalised; char ch; float lm, partial; } Node;
/* pyctcdecode's rules: an out-of-vocabulary word costs UNK_OFFSET in log10 units before alpha;
 * a partial word that is no vocabulary prefix costs UNK_OFFSET (raw), scaled up past 6 letters,
 * as a ranking term that is recomputed, not accumulated */
#define UNK_OFFSET (-10.0f)
#define LN10 2.302585093f
typedef struct { Node *n; int count, cap; int *map; long mapcap; } Tree;   /* map: (parent, class) -> child */

static int tree_child(Tree *t, int parent, int cls, const LM *lm, float alpha, float beta) {
    long key = (long)parent * 32 + cls;
    if ((long)t->count * 2 >= t->mapcap) {                          /* grow and rehash */
        long nc = t->mapcap * 2; int *nm = malloc(nc * sizeof(int)); for (long i = 0; i < nc; i++) nm[i] = -1;
        for (int i = 1; i < t->count; i++) {
            long k = (long)t->n[i].parent * 32 + (t->n[i].ch == '\'' ? 1 : t->n[i].ch == ' ' ? 2 : t->n[i].ch - 'A' + 3);
            long h = (k * 2654435761u) & (nc - 1); while (nm[h] >= 0) h = (h + 1) & (nc - 1); nm[h] = i;
        }
        free(t->map); t->map = nm; t->mapcap = nc;
    }
    long h = (key * 2654435761u) & (t->mapcap - 1);
    while (t->map[h] >= 0) {
        int i = t->map[h];
        long k = (long)t->n[i].parent * 32 + (t->n[i].ch == '\'' ? 1 : t->n[i].ch == ' ' ? 2 : t->n[i].ch - 'A' + 3);
        if (k == key) return i;
        h = (h + 1) & (t->mapcap - 1);
    }
    if (t->count == t->cap) { t->cap *= 2; t->n = realloc(t->n, t->cap * sizeof(Node)); }
    int id = t->count++;
    Node *pn = &t->n[parent], *c = &t->n[id];
    *c = *pn; c->parent = parent; c->depth = pn->depth + 1;
    c->ch = cls == 1 ? '\'' : cls == 2 ? ' ' : 'A' + (cls - 3);
    char word[64]; int wl = 0;
    if (c->ch == ' ') {                     /* a word ends: score it */
        int i = parent; while (i > 0 && t->n[i].ch != ' ' && wl < 63) { word[wl++] = t->n[i].ch; i = t->n[i].parent; }
        for (int a = 0, b = wl - 1; a < b; a++, b--) { char x = word[a]; word[a] = word[b]; word[b] = x; }
        int w = lm_id(lm, word, wl);
        c->lm += alpha * (lm_score(lm, pn->w1, pn->w2, w >= 0 ? w : lm->unk) + (w < 0 ? UNK_OFFSET * LN10 : 0)) + beta;
        c->w1 = pn->w2; c->w2 = w >= 0 ? w : lm->unk; c->penalised = 0; c->wstart = c->depth; c->partial = 0;
    } else {                                /* a word grows: is it still a prefix of some word? */
        int i = id; while (i > 0 && t->n[i].ch != ' ' && wl < 63) { word[wl++] = t->n[i].ch; i = t->n[i].parent; }
        for (int a = 0, b = wl - 1; a < b; a++, b--) { char x = word[a]; word[a] = word[b]; word[b] = x; }
        c->partial = lm_is_prefix(lm, word, wl) ? 0 : UNK_OFFSET * (wl > 6 ? wl / 6.0f : 1.0f);
    }
    t->map[h] = id;
    return id;
}

static float lse2(float a, float b) { if (a < b) { float x = a; a = b; b = x; } return b < -1e29f ? a : a + log1pf(expf(b - a)); }

typedef struct { int node; float pb, pnb; } Hyp;

/* decode logp [frames][classes]; writes the best text (no leading/trailing spaces) to out */
static void beam_decode(const float *logp, int frames, int classes, const LM *lm, float alpha, float beta, int B, char *out, int outcap) {
    Tree t; t.cap = 1 << 16; t.n = malloc(t.cap * sizeof(Node)); t.count = 1; t.mapcap = 1 << 17;
    t.map = malloc(t.mapcap * sizeof(int)); for (long i = 0; i < t.mapcap; i++) t.map[i] = -1;
    memset(&t.n[0], 0, sizeof(Node)); t.n[0].parent = -1; t.n[0].w1 = -1; t.n[0].w2 = lm->bos; t.n[0].ch = ' ';
    Hyp *cur = malloc(B * sizeof(Hyp)), *cand = malloc((size_t)B * (classes + 1) * sizeof(Hyp));
    int ncur = 1; cur[0].node = 0; cur[0].pb = 0; cur[0].pnb = -1e30f;
    int slotcap = 1; while (slotcap < 4 * B * (classes + 1)) slotcap <<= 1;
    int *slot = malloc(slotcap * sizeof(int));
    for (int f = 0; f < frames; f++) {
        const float *lp = logp + (size_t)f * classes;
        float mx = lp[0]; for (int c = 1; c < classes; c++) if (lp[c] > mx) mx = lp[c];
        for (int i = 0; i < slotcap; i++) slot[i] = -1;
        int nc = 0;
#define ADD(nodeid, which, val) do { int _n = (nodeid); long _h = ((long)_n * 2654435761u) & (slotcap - 1); \
        while (slot[_h] >= 0 && cand[slot[_h]].node != _n) _h = (_h + 1) & (slotcap - 1); \
        if (slot[_h] < 0) { slot[_h] = nc; cand[nc].node = _n; cand[nc].pb = -1e30f; cand[nc].pnb = -1e30f; nc++; } \
        cand[slot[_h]].which = lse2(cand[slot[_h]].which, (val)); } while (0)
        for (int b = 0; b < ncur; b++) {
            Hyp *h = &cur[b]; Node *nd = &t.n[h->node];
            float p = lse2(h->pb, h->pnb);
            int lastcls = h->node == 0 ? 0 : (nd->ch == '\'' ? 1 : nd->ch == ' ' ? 2 : nd->ch - 'A' + 3);
            ADD(h->node, pb, p + lp[0]);                                     /* blank */
            if (lastcls > 0) ADD(h->node, pnb, h->pnb + lp[lastcls]);        /* repeat collapses */
            for (int c = 1; c < classes; c++) {
                if (lp[c] < mx - 12.0f) continue;                            /* hopeless classes */
                if (c == 2 && (h->node == 0 || nd->ch == ' ')) { ADD(h->node, pnb, p + lp[c]); continue; }   /* no empty words */
                int child = tree_child(&t, h->node, c, lm, alpha, beta);
                nd = &t.n[h->node];                                          /* tree may have moved */
                ADD(child, pnb, (c == lastcls ? h->pb : p) + lp[c]);
            }
        }
#undef ADD
        /* keep the B best by acoustic + LM score */
        for (int i = 0; i < nc; i++) for (int j = i + 1; j < nc && i < B; j++) {
            float si = lse2(cand[i].pb, cand[i].pnb) + t.n[cand[i].node].lm + t.n[cand[i].node].partial;
            float sj = lse2(cand[j].pb, cand[j].pnb) + t.n[cand[j].node].lm + t.n[cand[j].node].partial;
            if (sj > si) { Hyp x = cand[i]; cand[i] = cand[j]; cand[j] = x; }
        }
        ncur = nc < B ? nc : B;
        memcpy(cur, cand, ncur * sizeof(Hyp));
    }
    /* finish: score the last partial word and the end of sentence */
    float best = -1e30f; int bestn = 0;
    for (int b = 0; b < ncur; b++) {
        int n = cur[b].node; Node *nd = &t.n[n];
        float s = lse2(cur[b].pb, cur[b].pnb) + nd->lm;
        int w1 = nd->w1, w2 = nd->w2;
        if (n != 0 && nd->ch != ' ') {
            char word[64]; int wl = 0, i = n;
            while (i > 0 && t.n[i].ch != ' ' && wl < 63) { word[wl++] = t.n[i].ch; i = t.n[i].parent; }
            for (int a = 0, c = wl - 1; a < c; a++, c--) { char x = word[a]; word[a] = word[c]; word[c] = x; }
            int w = lm_id(lm, word, wl);
            s += alpha * (lm_score(lm, w1, w2, w >= 0 ? w : lm->unk) + (w < 0 ? UNK_OFFSET * LN10 : 0)) + beta;
            w1 = w2; w2 = w >= 0 ? w : lm->unk;
        }
        s += alpha * lm_score(lm, w1, w2, lm->eos);
        if (s > best) { best = s; bestn = n; }
    }
    int len = t.n[bestn].depth; if (len >= outcap) len = outcap - 1;
    out[len] = 0;
    for (int i = bestn, k = t.n[bestn].depth - 1; i > 0; i = t.n[i].parent, k--) if (k < len) out[k] = t.n[i].ch;
    char *a = out; while (*a == ' ') a++;
    memmove(out, a, strlen(a) + 1);
    for (int k = (int)strlen(out) - 1; k >= 0 && out[k] == ' '; k--) out[k] = 0;
    free(t.n); free(t.map); free(cur); free(cand); free(slot);
}

/* test mode: decode a file of dumped log-probabilities (int32 count; per utterance int32 frames + floats) */
static int decode_logp_file(const char *path, const LM *lm, float alpha, float beta, int B) {
    FILE *f = fopen(path, "rb"); if (!f) { perror(path); return 1; }
    int count; if (fread(&count, 4, 1, f) != 1) return 1;
    char *out = malloc(1 << 16);
    for (int u = 0; u < count; u++) {
        int frames; if (fread(&frames, 4, 1, f) != 1) return 1;
        float *lp = malloc((size_t)frames * 29 * 4);
        if (fread(lp, 4, (size_t)frames * 29, f) != (size_t)frames * 29) return 1;
        beam_decode(lp, frames, 29, lm, alpha, beta, B, out, 1 << 16);
        printf("%s\n", out); fflush(stdout); free(lp);
    }
    return 0;
}


/* --stream: raw signed 16-bit mono 16 kHz on stdin, e.g.
 *   arecord -q -D plughw:1,0 -f S16_LE -r 16000 -c 1 -t raw | zara_stt MODEL.bin --stream --int8 --lm LM.zlm
 * Letters appear as the model commits to them (greedy, every K frames). A phrase ends when the
 * model has emitted only blanks for PAUSE_FRAMES after some text; with --lm the phrase is then
 * re-decoded by the beam search and its line replaced by the result. A running RMS estimate
 * (2 s window, floor 0.003) scales the input to RMS 0.1, as the training did per utterance. */
#define PAUSE_FRAMES 35
/* --mic: stdin is the uConsole headset mic as the ES8388 delivers it, stereo S16 at 44.1 kHz with
 * the differential mic as L and an inverted R (their sum cancels the voice). We take (L - R) / 2
 * and resample to 16 kHz with a Hann-windowed sinc (cutoff 7.6 kHz, 32 taps), dropping the first
 * half second (the ADC power-up thump). */
static int g_mic = 0;
#define MIC_RATE 44100.0
#define MIC_HALF 16
static float *mic_in; static long mic_cap, mic_len, mic_base; static double mic_pos; static long mic_skip = 8000;
static int mic_fill(long upto) {            /* make input sample `upto` (absolute) available */
    while (mic_base + mic_len <= upto) {
        int16_t st[2 * 1024]; size_t got = fread(st, 4, 1024, stdin);
        if (got == 0) return 0;
        if (mic_len + (long)got > mic_cap) {
            long keep = (long)mic_pos - MIC_HALF - 1 - mic_base;      /* drop history we no longer need */
            if (keep > 0) { memmove(mic_in, mic_in + keep, (mic_len - keep) * 4); mic_len -= keep; mic_base += keep; }
            if (mic_len + (long)got > mic_cap) { mic_cap = (mic_len + got) * 2; mic_in = realloc(mic_in, mic_cap * 4); }
        }
        for (size_t i = 0; i < got; i++) mic_in[mic_len++] = (st[2 * i] - st[2 * i + 1]) / 65536.0f;
    }
    return 1;
}
static int mic_sample(float *y) {           /* the next 16 kHz sample */
    for (;;) {
        long c = (long)floor(mic_pos);
        if (!mic_fill(c + MIC_HALF)) return 0;
        const double fc = 7600.0 / MIC_RATE;                       /* cutoff, cycles per input sample */
        double acc = 0;
        for (long i = c - MIC_HALF + 1; i <= c + MIC_HALF; i++) {
            if (i < mic_base) continue;
            double d = mic_pos - i, x = 2 * fc * d;
            double sinc = fabs(x) < 1e-9 ? 1.0 : sin(M_PI * x) / (M_PI * x);
            double win = 0.5 + 0.5 * cos(M_PI * d / MIC_HALF);
            acc += mic_in[i - mic_base] * 2 * fc * sinc * win;
        }
        mic_pos += MIC_RATE / 16000.0;
        if (mic_skip > 0) { mic_skip--; continue; }
        *y = (float)acc; return 1;
    }
}
static int stream(Model *m, const LM *lm, float alpha, float beta, int B, int K) {
    int frame = m->h.frame, classes = m->h.classes, tty = isatty(1);
    int16_t *buf = malloc((size_t)frame * 2);
    size_t segcap = 1 << 12, segn = 0; float *seg = malloc(segcap * classes * 4);
    char *line = malloc(1 << 16), *out = malloc(1 << 16); int llen = 0, last = 0, blanks = 0;
    float *lp = malloc((size_t)K * classes * 4);
    double rms2 = 1e-4, a = 1.0 / (2.0 * m->h.rate);
    chunk_alloc(m, K);
    for (;;) {
        int kn = 0, eof = 0;
        for (; kn < K; kn++) {
            if (g_mic) {
                int ok = 1;
                for (int i = 0; i < frame && ok; i++) { float y = 0; ok = mic_sample(&y); buf[i] = (int16_t)fmaxf(-32768.f, fminf(32767.f, y * 32768.f)); }
                if (!ok) { eof = 1; break; }
            } else if (fread(buf, 2, frame, stdin) != (size_t)frame) { eof = 1; break; }
            float *fk = m->cFEAT + (size_t)kn * m->h.nin;
            for (int i = 0; i < frame; i++) {
                float x = buf[i] / 32768.0f;
                rms2 += a * ((double)x * x - rms2);
                float r = sqrtf((float)rms2); if (r < 0.003f) r = 0.003f;
                x *= 0.1f / r;
                if (m->h.front == 0) bank_sample(m, x); else mel_sample(m, x);
            }
            long t = (long)(segn + kn);   /* the mel front only needs a monotone frame index */
            if (m->h.front == 0) bank_frame(m, fk); else mel_frame(m, t, (t + 1) * frame, fk);
        }
        if (kn > 0) {
            frames_chunk(m, kn, lp);
            for (int k = 0; k < kn; k++) {
                if (segn == segcap) { segcap *= 2; seg = realloc(seg, segcap * classes * 4); }
                memcpy(seg + segn * classes, lp + (size_t)k * classes, classes * 4); segn++;
                const float *l = lp + (size_t)k * classes;
                int best = 0; for (int c = 1; c < classes; c++) if (l[c] > l[best]) best = c;
                if (best != last && best != 0) {
                    char ch = best == 1 ? '\'' : best == 2 ? ' ' : 'A' + (best - 3);
                    if (!(ch == ' ' && (llen == 0 || line[llen - 1] == ' '))) { line[llen++] = ch; putchar(ch); }
                }
                blanks = best == 0 ? blanks + 1 : 0;
                last = best;
            }
            fflush(stdout);
        }
        int pause = llen > 0 && blanks >= PAUSE_FRAMES;
        if (pause || (eof && llen > 0)) {
            if (lm) {
                beam_decode(seg, (int)segn, classes, lm, alpha, beta, B, out, 1 << 16);
                if (tty) printf("\r\033[K%s\n", out); else printf("\n= %s\n", out);
            } else putchar('\n');
            fflush(stdout);
            llen = 0; segn = 0; blanks = 0;
        } else if (blanks >= PAUSE_FRAMES && llen == 0) segn = 0;   /* silence before any text: drop it */
        if (eof) break;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s MODEL.bin FILE.wav [--dump OUT.f32] [--bench] [--gain G]\n", argv[0]); return 1; }
    const char *dump = NULL, *lmpath = NULL, *logpfile = NULL; int bench = 0, beam = 32; float gain = 0, alpha = 0.5f, beta = 0.5f; int K = 16;
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--dump") && i + 1 < argc) dump = argv[++i];
        else if (!strcmp(argv[i], "--bench")) bench = 1;
        else if (!strcmp(argv[i], "--gain") && i + 1 < argc) gain = atof(argv[++i]);
        else if (!strcmp(argv[i], "--chunk") && i + 1 < argc) K = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--lm") && i + 1 < argc) lmpath = argv[++i];
        else if (!strcmp(argv[i], "--alpha") && i + 1 < argc) alpha = atof(argv[++i]);
        else if (!strcmp(argv[i], "--beta") && i + 1 < argc) beta = atof(argv[++i]);
        else if (!strcmp(argv[i], "--beam") && i + 1 < argc) beam = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--int8")) g_int8 = 1;
        else if (!strcmp(argv[i], "--mic")) g_mic = 1;
        else { fprintf(stderr, "unknown option %s\n", argv[i]); return 1; }
    }
    if (!strcmp(argv[1], "--decode-logp")) {          /* zara_stt --decode-logp FILE --lm LM.bin [...]: decoder test */
        logpfile = argv[2];
        if (!lmpath) { fprintf(stderr, "--decode-logp needs --lm\n"); return 1; }
        return decode_logp_file(logpfile, load_lm(lmpath), alpha, beta, beam);
    }
    Model *m = load_model(argv[1]);
    LM *lm = lmpath ? load_lm(lmpath) : NULL;
    if (!strcmp(argv[2], "--stream")) return stream(m, lm, alpha, beta, beam, K < 1 ? 1 : K);
    size_t n; float *x = read_wav(argv[2], &n, m->h.rate);
    if (gain == 0) {
        double s = 0; for (size_t i = 0; i < n; i++) s += (double)x[i] * x[i];
        gain = 0.1f / (sqrtf(s / (n ? n : 1)) + 1e-5f);
    }
    int frame = m->h.frame, classes = m->h.classes;
    size_t frames = (n + frame - 1) / frame;
    float *feat = malloc(m->h.nin * 4), *logp = malloc(frames * classes * 4);
    char *text = malloc(frames + 1); int tlen = 0, last = 0;
    if (K < 1) K = 1;
    chunk_alloc(m, K);
    double t0 = now();
    for (size_t t0f = 0; t0f < frames; t0f += K) {
        int kn = (int)((frames - t0f) < (size_t)K ? frames - t0f : (size_t)K);
        for (int k = 0; k < kn; k++) {
            size_t t = t0f + k; float *fk = m->cFEAT + (size_t)k * m->h.nin;
            if (m->h.front == 0) {
                for (int i = 0; i < frame; i++) { size_t j = t * frame + i; bank_sample(m, j < n ? x[j] * gain : 0); }
                bank_frame(m, fk);
            } else {
                for (int i = 0; i < frame; i++) { size_t j = t * frame + i; mel_sample(m, j < n ? x[j] * gain : 0); }
                mel_frame(m, (long)t, (long)(t + 1) * frame, fk);
            }
        }
        frames_chunk(m, kn, logp + t0f * classes);
        for (int k = 0; k < kn; k++) {
            float *lp = logp + (t0f + k) * classes;
            int best = 0; for (int c = 1; c < classes; c++) if (lp[c] > lp[best]) best = c;
            if (best != last && best != 0) text[tlen++] = best == 1 ? '\'' : best == 2 ? ' ' : 'A' + (best - 3);
            last = best;
        }
    }
    (void)feat; (void)frame_step;
    double t1 = now(), tlm = t1;
    text[tlen] = 0;
    if (lm) {
        char *out = malloc(1 << 16);
        beam_decode(logp, (int)frames, classes, lm, alpha, beta, beam, out, 1 << 16);
        tlm = now();
        printf("%s\n", out);
    } else printf("%s\n", text);
    if (dump) { FILE *f = fopen(dump, "wb"); fwrite(logp, 4, frames * classes, f); fclose(f); }
    if (bench) {
        double audio = (double)n / m->h.rate, wall = t1 - t0;
        fprintf(stderr, "audio %.2f s, compute %.3f s, real-time factor %.4f (%.1fx faster than real time)\n", audio, wall, wall / audio, audio / wall);
        if (lm) fprintf(stderr, "beam search with LM: %.3f s, real-time factor %.4f\n", tlm - t1, (tlm - t1) / audio);
    }
    return 0;
}
