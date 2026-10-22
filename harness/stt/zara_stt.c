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

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s MODEL.bin FILE.wav [--dump OUT.f32] [--bench] [--gain G]\n", argv[0]); return 1; }
    const char *dump = NULL; int bench = 0; float gain = 0;
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--dump") && i + 1 < argc) dump = argv[++i];
        else if (!strcmp(argv[i], "--bench")) bench = 1;
        else if (!strcmp(argv[i], "--gain") && i + 1 < argc) gain = atof(argv[++i]);
        else { fprintf(stderr, "unknown option %s\n", argv[i]); return 1; }
    }
    Model *m = load_model(argv[1]);
    size_t n; float *x = read_wav(argv[2], &n, m->h.rate);
    if (gain == 0) {
        double s = 0; for (size_t i = 0; i < n; i++) s += (double)x[i] * x[i];
        gain = 0.1f / (sqrtf(s / (n ? n : 1)) + 1e-5f);
    }
    int frame = m->h.frame, classes = m->h.classes;
    size_t frames = (n + frame - 1) / frame;
    float *feat = malloc(m->h.nin * 4), *logp = malloc(frames * classes * 4);
    char *text = malloc(frames + 1); int tlen = 0, last = 0;
    double t0 = now();
    for (size_t t = 0; t < frames; t++) {
        if (m->h.front == 0) {
            for (int i = 0; i < frame; i++) { size_t j = t * frame + i; bank_sample(m, j < n ? x[j] * gain : 0); }
            bank_frame(m, feat);
        } else {
            for (int i = 0; i < frame; i++) { size_t j = t * frame + i; mel_sample(m, j < n ? x[j] * gain : 0); }
            mel_frame(m, (long)t, (long)(t + 1) * frame, feat);
        }
        float *lp = logp + t * classes;
        frame_step(m, feat, lp);
        int best = 0; for (int c = 1; c < classes; c++) if (lp[c] > lp[best]) best = c;
        if (best != last && best != 0) text[tlen++] = best == 1 ? '\'' : best == 2 ? ' ' : 'A' + (best - 3);
        last = best;
    }
    double t1 = now();
    text[tlen] = 0;
    printf("%s\n", text);
    if (dump) { FILE *f = fopen(dump, "wb"); fwrite(logp, 4, frames * classes, f); fclose(f); }
    if (bench) {
        double audio = (double)n / m->h.rate, wall = t1 - t0;
        fprintf(stderr, "audio %.2f s, compute %.3f s, real-time factor %.4f (%.1fx faster than real time)\n", audio, wall, wall / audio, audio / wall);
    }
    return 0;
}
