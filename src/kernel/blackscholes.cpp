#include "blackscholes.h"
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>

// Kept text-identical to course starter for naive / CNDF (float promotion rules).
#define inv_sqrt_2xPI 0.39894228040143270286
#define p_val 0.2316419
#define coefficient_a1 0.319381530
#define coefficient_a2 -0.356563782
#define coefficient_a3 1.781477937
#define coefficient_a4 -1.821255978
#define coefficient_a5 1.330274429

// ---- Student path: LUTs + local approximations (not used by naive) ---------
namespace {
constexpr int GAUSS_N = 2048; // segments in [0, GAUSS_X_MAX]
constexpr float GAUSS_X_MAX = 10.0f;
constexpr int LOGN = 512; // log LUT segments for ratio s/k
constexpr float R_LO = 0.3f;
constexpr float R_HI = 3.0f;

static std::array<float, GAUSS_N + 1> make_gauss() {
    std::array<float, GAUSS_N + 1> a{};
    for (int i = 0; i <= GAUSS_N; ++i) {
        const float x = (static_cast<float>(i) * GAUSS_X_MAX) /
                         static_cast<float>(GAUSS_N);
        a[static_cast<std::size_t>(i)] = std::exp(-0.5f * x * x);
    }
    return a;
}

static std::array<float, LOGN + 1> make_lograt() {
    std::array<float, LOGN + 1> a{};
    for (int i = 0; i <= LOGN; ++i) {
        const float r =
            R_LO + (R_HI - R_LO) * (static_cast<float>(i) / static_cast<float>(LOGN));
        a[static_cast<std::size_t>(i)] = std::log(r);
    }
    return a;
}

static const std::array<float, GAUSS_N + 1> kGaussLut = make_gauss();
static const std::array<float, LOGN + 1> kLogRatLut = make_lograt();

__attribute__((always_inline)) inline float gauss_nprime_lerp(float x) {
    if (x >= GAUSS_X_MAX) {
        return 0.0f;
    }
    const float s = (x / GAUSS_X_MAX) * static_cast<float>(GAUSS_N);
    int i = static_cast<int>(s);
    if (i < 0) {
        i = 0;
    }
    if (i >= GAUSS_N) {
        return 0.0f;
    }
    const float f = s - static_cast<float>(i);
    const float g0 = kGaussLut[static_cast<std::size_t>(i)];
    const float g1 = kGaussLut[static_cast<std::size_t>(i) + 1U];
    return fmaf(f, g1 - g0, g0);
}

// log(s/k) for ratio in a tight band: LUT; otherwise fall back to libm
__attribute__((always_inline)) inline float log_ratio_stu(float ratio) {
    if (ratio < R_LO || ratio > R_HI) {
        return std::log(ratio);
    }
    const float t =
        (ratio - R_LO) / (R_HI - R_LO) * static_cast<float>(LOGN);
    int i = static_cast<int>(t);
    if (i < 0) {
        i = 0;
    }
    if (i >= LOGN) {
        return kLogRatLut[static_cast<std::size_t>(LOGN)];
    }
    const float f = t - static_cast<float>(i);
    const float y0 = kLogRatLut[static_cast<std::size_t>(i)];
    const float y1 = kLogRatLut[static_cast<std::size_t>(i) + 1U];
    return fmaf(f, y1 - y0, y0);
}

// exp(-r t) with u = -r t ∈ [-0.1,0]; 7th-order Taylor (no lib expf on hot path)
__attribute__((always_inline)) inline float exp_neg_rt(float r, float t) {
    const float u = -(r) * t;
    const float u2 = u * u;
    const float u3 = u2 * u;
    const float u4 = u3 * u;
    const float u5 = u4 * u;
    const float u6 = u5 * u;
    const float u7 = u6 * u;
    return 1.0f + u + 0.5f * u2 + (1.0f / 6.0f) * u3 + (1.0f / 24.0f) * u4 +
           (1.0f / 120.0f) * u5 + (1.0f / 720.0f) * u6 + (1.0f / 5040.0f) * u7;
}

__attribute__((always_inline)) inline float stu_cndf(float input_x) {
    int sign = 0;
    float x = input_x;
    if (x < 0.0f) {
        x = -x;
        sign = 1;
    }
    const float xNPrimeofX = gauss_nprime_lerp(x) * static_cast<float>(inv_sqrt_2xPI);
    const float k = 1.0f / (1.0f + static_cast<float>(p_val) * x);
    const float k_2 = k * k;
    const float k_3 = k_2 * k;
    const float k_4 = k_3 * k;
    const float k_5 = k_4 * k;
    float local = k * static_cast<float>(coefficient_a1);
    local += k_2 * static_cast<float>(coefficient_a2);
    local += k_3 * static_cast<float>(coefficient_a3);
    local += k_4 * static_cast<float>(coefficient_a4);
    local += k_5 * static_cast<float>(coefficient_a5);
    local = 1.0f - local * xNPrimeofX;
    return sign ? (1.0f - local) : local;
}
} // namespace

void initialize_blackscholes(blackscholes_args &args,
                             std::size_t n,
                             std::uint32_t seed) {
    args.call_option_price.assign(n, 0.0f);
    args.put_option_price.assign(n, 0.0f);
    args.epsilon = 5e-3;

    args.spot_price.resize(n);
    args.strike.resize(n);
    args.rate.resize(n);
    args.volatility.resize(n);
    args.time.resize(n);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> spot_dist(50.0f, 99.9f);
    std::uniform_real_distribution<float> strike_dist(50.0f, 99.9f);
    std::uniform_real_distribution<float> rate_dist(0.0275f, 0.1f);
    std::uniform_real_distribution<float> vol_dist(0.05f, 0.6f);
    std::uniform_real_distribution<float> time_dist(0.1f, 1.0f);

    for (std::size_t i = 0; i < n; ++i) {
        args.spot_price[i] = spot_dist(rng);
        args.strike[i] = strike_dist(rng);
        args.rate[i] = rate_dist(rng);
        args.volatility[i] = vol_dist(rng);
        args.time[i] = time_dist(rng);
    }
}

void CNDF(float &InputX, float &OutputX) {
    int sign = 0;
    float x = InputX;

    if (x < 0.0f) {
        x = -x;
        sign = 1;
    }

    const float xNPrimeofX = std::exp(-0.5f * x * x) * inv_sqrt_2xPI;
    const float k = 1.0f / (1.0f + p_val * x);
    const float k_2 = k * k;
    const float k_3 = k_2 * k;
    const float k_4 = k_3 * k;
    const float k_5 = k_4 * k;

    float local = k * coefficient_a1;
    local += k_2 * coefficient_a2;
    local += k_3 * coefficient_a3;
    local += k_4 * coefficient_a4;
    local += k_5 * coefficient_a5;
    local = 1.0f - local * xNPrimeofX;

    OutputX = sign ? (1.0f - local) : local;
}

static inline void naive_BlkSchls_one(float &CallOptionPrice,
                                      float &PutOptionPrice, float spotPrice,
                                      float strike, float rate,
                                      float volatility, float time) {
    const float xSqrtTime = std::sqrt(time);
    const float xLogTerm = std::log(spotPrice / strike);
    const float xPowerTerm = 0.5f * volatility * volatility;

    float xD1 = (rate + xPowerTerm) * time + xLogTerm;
    const float xDen = volatility * xSqrtTime;
    xD1 = xD1 / xDen;
    const float xD2 = xD1 - xDen;

    float d1 = xD1;
    float d2 = xD2;
    float NofXd1 = 0.0f;
    float NofXd2 = 0.0f;

    CNDF(d1, NofXd1);
    CNDF(d2, NofXd2);

    const float FutureValueX = strike * std::exp(-(rate) * (time));
    CallOptionPrice = (spotPrice * NofXd1) - (FutureValueX * NofXd2);

    const float NegNofXd1 = 1.0f - NofXd1;
    const float NegNofXd2 = 1.0f - NofXd2;
    PutOptionPrice = (FutureValueX * NegNofXd2) - (spotPrice * NegNofXd1);
}

void naive_BlkSchls(std::vector<float> &CallOptionPrice,
                    std::vector<float> &PutOptionPrice,
                    const std::vector<float> &spotPrice,
                    const std::vector<float> &strike,
                    const std::vector<float> &rate,
                    const std::vector<float> &volatility,
                    const std::vector<float> &time) {
    size_t n = spotPrice.size();
    for (size_t i = 0; i < n; ++i) {
        naive_BlkSchls_one(CallOptionPrice[i],
                           PutOptionPrice[i],
                           spotPrice[i],
                           strike[i],
                           rate[i],
                           volatility[i],
                           time[i]);
    }
}

void stu_BlkSchls(std::vector<float> &CallOptionPrice,
                  std::vector<float> &PutOptionPrice,
                  const std::vector<float> &spotPrice,
                  const std::vector<float> &strike,
                  const std::vector<float> &rate,
                  const std::vector<float> &volatility,
                  const std::vector<float> &time) {
    const size_t n = spotPrice.size();
    if (n == 0) {
        return;
    }
    const float *const __restrict__ sp = spotPrice.data();
    const float *const __restrict__ st = strike.data();
    const float *const __restrict__ ra = rate.data();
    const float *const __restrict__ vo = volatility.data();
    const float *const __restrict__ ti = time.data();
    float *const __restrict__ call = CallOptionPrice.data();
    float *const __restrict__ puto = PutOptionPrice.data();

    size_t i = 0;
    for (; i + 3 < n; i += 4) {
#if defined(__GNUC__) && !defined(__NuttX__)
        __builtin_prefetch(&sp[i + 8], 0, 1);
        __builtin_prefetch(&st[i + 8], 0, 1);
        __builtin_prefetch(&ra[i + 8], 0, 1);
        __builtin_prefetch(&vo[i + 8], 0, 1);
        __builtin_prefetch(&ti[i + 8], 0, 1);
#endif
        for (int j = 0; j < 4; ++j) {
            const size_t k = i + static_cast<std::size_t>(j);
            const float s = sp[k];
            const float k_strike = st[k];
            const float r = ra[k];
            const float v = vo[k];
            const float t = ti[k];

            const float xSqrtTime = std::sqrt(t);
            const float xLogTerm = log_ratio_stu(s / k_strike);
            const float xPowerTerm = 0.5f * v * v;
            float xD1 = (r + xPowerTerm) * t + xLogTerm;
            const float xDen = v * xSqrtTime;
            xD1 = xD1 / xDen;
            const float xD2 = xD1 - xDen;

            const float N1 = stu_cndf(xD1);
            const float N2 = stu_cndf(xD2);
            const float fv = k_strike * exp_neg_rt(r, t);
            const float c = (s * N1) - (fv * N2);
            const float p = (fv * (1.0f - N2)) - (s * (1.0f - N1));
            call[k] = c;
            puto[k] = p;
        }
    }
    for (; i < n; ++i) {
        const float s = sp[i];
        const float k_strike = st[i];
        const float r = ra[i];
        const float v = vo[i];
        const float t = ti[i];

        const float xSqrtTime = std::sqrt(t);
        const float xLogTerm = log_ratio_stu(s / k_strike);
        const float xPowerTerm = 0.5f * v * v;
        float xD1 = (r + xPowerTerm) * t + xLogTerm;
        const float xDen = v * xSqrtTime;
        xD1 = xD1 / xDen;
        const float xD2 = xD1 - xDen;

        const float N1 = stu_cndf(xD1);
        const float N2 = stu_cndf(xD2);
        const float fv = k_strike * exp_neg_rt(r, t);
        const float c = (s * N1) - (fv * N2);
        const float p = (fv * (1.0f - N2)) - (s * (1.0f - N1));
        call[i] = c;
        puto[i] = p;
    }
}

void naive_BlkSchls_wrapper(void *ctx) {
    auto &args = *static_cast<blackscholes_args *>(ctx);
    naive_BlkSchls(args.call_option_price,
                   args.put_option_price,
                   args.spot_price,
                   args.strike,
                   args.rate,
                   args.volatility,
                   args.time);
}

void stu_BlkSchls_wrapper(void *ctx) {
    auto &args = *static_cast<blackscholes_args *>(ctx);
    stu_BlkSchls(args.call_option_price,
                 args.put_option_price,
                 args.spot_price,
                 args.strike,
                 args.rate,
                 args.volatility,
                 args.time);
}

bool BlkSchls_check(void *stu_ctx, void *ref_ctx, lab_test_func naive_func) {
    naive_func(ref_ctx);
    auto &stu_args = *static_cast<blackscholes_args *>(stu_ctx);
    auto &ref_args = *static_cast<blackscholes_args *>(ref_ctx);
    const double eps = ref_args.epsilon; // relative tolerance

    if (ref_args.call_option_price.size() != stu_args.call_option_price.size() ||
        ref_args.put_option_price.size() != stu_args.put_option_price.size())
        return false;

    const double atol = 1e-5; // absolute tolerance for near-zero prices
    const size_t n = ref_args.call_option_price.size();
    double max_rel = 0.0, max_abs = 0.0;
    size_t max_idx = 0;
    const char *max_leg = "call";

    for (size_t i = 0; i < n; ++i) {
        const double rc = static_cast<double>(ref_args.call_option_price[i]);
        const double rp = static_cast<double>(ref_args.put_option_price[i]);
        const double sc = static_cast<double>(stu_args.call_option_price[i]);
        const double sp = static_cast<double>(stu_args.put_option_price[i]);

        const double err_c = std::abs(rc - sc);
        const double err_p = std::abs(rp - sp);
        const double rel_c = (err_c - atol) / std::abs(rc);
        const double rel_p = (err_p - atol) / std::abs(rp);

        const bool call_ok = err_c <= (atol + eps * std::abs(rc));
        const bool put_ok = err_p <= (atol + eps * std::abs(rp));

        if (rel_c > max_rel) {
            max_abs = err_c;
            max_rel = rel_c;
            max_idx = i;
            max_leg = "call";
        }
        if (rel_p > max_rel) {
            max_abs = err_p;
            max_rel = rel_p;
            max_idx = i;
            max_leg = "put";
        }

        if (!call_ok || !put_ok) {
            debug_log("\tDEBUG: fail idx={} | call ref={} stu={} err={} thr={} | put ref={} stu={} err={} thr={}\n",
                      i,
                      rc,
                      sc,
                      err_c,
                      (atol + eps * std::abs(rc)),
                      rp,
                      sp,
                      err_p,
                      (atol + eps * std::abs(rp)));
            return false;
        }
    }
    debug_log("\tBlkSchls_check passed: n={}, max_rel_err={}, max_abs_err={} at idx={} ({})\n",
              n,
              max_rel,
              max_abs,
              max_idx,
              max_leg);

    return true;
}
