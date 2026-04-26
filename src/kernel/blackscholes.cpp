#include "blackscholes.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>

#define inv_sqrt_2xPI 0.39894228040143270286f
#define p_val 0.2316419f
#define coefficient_a1 0.319381530f
#define coefficient_a2 -0.356563782f
#define coefficient_a3 1.781477937f
#define coefficient_a4 -1.821255978f
#define coefficient_a5 1.330274429f

// ---- Student-only fast helpers (not used by naive reference) ----
namespace {
// log2(1 + t) for t in [0, 0.4142] (mantissa after [1,sqrt(2)) renormalization)
__attribute__((always_inline)) inline float
stu_log2_1p_small(float t) {
    // ln(1+t) via Taylor, then * (1/ln(2)) = * 1.4426950408
    const float t2 = t * t;
    const float t3 = t2 * t;
    const float t4 = t3 * t;
    const float t5 = t4 * t;
    // ln(1+t); then log2(1+t) = ln(1+t) * (1/ln(2))
    const float ln1p = fmaf(0.2f, t5, fmaf(-0.25f, t4, fmaf(0.333333343f, t3, fmaf(-0.5f, t2, t))));
    return 1.4426950408f * ln1p;
}

// log2 for positive float (Abramowitz-style: exponent + log2(1+small))
__attribute__((always_inline)) inline float stu_flog2(float x) {
    std::uint32_t bits = std::bit_cast<std::uint32_t>(x);
    int e = int(bits >> 23) - 127;
    bits = (bits & 0x007fffffu) | 0x3f800000u;
    float m = std::bit_cast<float>(bits);
    if (m > 1.4142135f) {
        m *= 0.5f;
        ++e;
    }
    const float t = m - 1.0f;
    return static_cast<float>(e) + stu_log2_1p_small(t);
}

// 2^f = exp(f ln 2), f in [0, 1) → g in [0, ln 2] ⊂ [0, 0.7]
__attribute__((always_inline)) inline float stu_pow2f_frac(float f) {
    const float g = f * 0.69314718056f;
    const float g2 = g * g;
    const float g3 = g2 * g;
    const float g4 = g3 * g;
    const float g5 = g4 * g;
    const float g6 = g5 * g;
    const float g7 = g6 * g;
    return 1.0f + g + 0.5f * g2 + g3 * (1.0f / 6.0f) + g4 * (1.0f / 24.0f) +
           g5 * (1.0f / 120.0f) + g6 * (1.0f / 720.0f) + g7 * (1.0f / 5040.0f);
}

// 2^a, full range (replaces std::exp without libm in hot path; ldexpf applies exact power-of-2)
__attribute__((always_inline)) inline float stu_fexp2(float a) {
    if (a <= -150.f) {
        return 0.0f;
    }
    if (a >= 128.f) {
        return 0x1.fffffep+127f;
    }
    const float fl = std::floor(a);
    const int n = static_cast<int>(fl);
    const float f = a - fl;
    const float p = stu_pow2f_frac(f);
    return std::ldexpf(p, n);
}

// exp(x) = exp2(x * log2(e))
__attribute__((always_inline)) inline float stu_fexp(float x) {
    return stu_fexp2(x * 1.44269504088896f);
}

// t ∈ [0.1,1] for this task — one lib sqrt is still cheap vs exp/log; keeps numerics
__attribute__((always_inline)) inline float stu_fsqrt(float x) { return std::sqrt(x); }

// Cumulative normal (same structure as reference CNDF, student intrinsics for exp)
__attribute__((always_inline)) inline float
stu_cndf_scalar(float input_x) {
    int sign = 0;
    float x = input_x;
    if (x < 0.0f) {
        x = -x;
        sign = 1;
    }
    const float xNPrimeofX = stu_fexp(-0.5f * x * x) * inv_sqrt_2xPI;
    const float k = 1.0f / (1.0f + p_val * x);
    const float k_2 = k * k;
    const float k_3 = k_2 * k;
    const float k_4 = k_3 * k;
    const float k_5 = k_4 * k;
    float local = k * coefficient_a1;
    local = fmaf(k_2, coefficient_a2, local);
    local = fmaf(k_3, coefficient_a3, local);
    local = fmaf(k_4, coefficient_a4, local);
    local = fmaf(k_5, coefficient_a5, local);
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

    const float ln2 = 0.69314718056f;

    for (size_t i = 0; i < n; ++i) {
        const float s = sp[i];
        const float k_strike = st[i];
        const float r = ra[i];
        const float v = vo[i];
        const float t = ti[i];

        const float xSqrtTime = stu_fsqrt(t);
        const float xLogTerm = stu_flog2(s / k_strike) * ln2;
        const float xPowerTerm = 0.5f * v * v;

        float xD1 = (r + xPowerTerm) * t + xLogTerm;
        const float xDen = v * xSqrtTime;
        xD1 = xD1 / xDen;
        const float xD2 = xD1 - xDen;

        const float NofXd1 = stu_cndf_scalar(xD1);
        const float NofXd2 = stu_cndf_scalar(xD2);
        const float FutureValueX = k_strike * stu_fexp(-(r) * t);

        const float c = (s * NofXd1) - (FutureValueX * NofXd2);
        const float NegNofXd1 = 1.0f - NofXd1;
        const float NegNofXd2 = 1.0f - NofXd2;
        const float p = (FutureValueX * NegNofXd2) - (s * NegNofXd1);
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
