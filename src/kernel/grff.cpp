#include "grff.h"
#include <algorithm>
#include <cmath>
#include <random>

void initialize_grff(grff_args *args, const size_t size, const std::uint_fast64_t seed) {
    if (!args) return;

    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    args->a_features.resize(size);
    args->b_features.resize(size);
    args->c_features.resize(size);
    args->f_output.resize(size);

    for (size_t i = 0; i < size; ++i) {
        args->a_features[i] = dist(gen);
        args->b_features[i] = dist(gen);
        args->c_features[i] = dist(gen);
    }
}

// -------------------------------------------------------------------------
// Naive Implementation (A Simplified Gated Residual Feature Fusion (GRFF))
// -------------------------------------------------------------------------
void naive_grff(grff_args& args) {
    size_t n = args.a_features.size();
    
    // Intermediate buffers
    std::vector<float> G(n), A_prime(n), Smooth_A(n), B_prime(n), C_prime(n), H(n), E(n);

    // Stage 1: Gate
    for (size_t i = 0; i < n; ++i) 
        G[i] = 0.5f * ((args.a_features[i] * args.b_features[i]) / (1.0f + std::abs(args.a_features[i] * args.b_features[i])) + 1.0f);

    // Stage 2: Update A (Residual)
    for (size_t i = 0; i < n; ++i) 
        A_prime[i] = args.a_features[i] + G[i];

    // Stage 3: Global Feature Scaling
    float sum_a = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum_a += A_prime[i];
    }
    float avg_a = sum_a / static_cast<float>(n);

    // Stage 4: Update A (Smooth)
    Smooth_A[0] = A_prime[0];
    for (size_t i = 1; i < n; ++i) {
        Smooth_A[i] = (A_prime[i] + A_prime[i-1]) * 0.5f; 
    }

    // Stage 5: Update B (Suppression)
    for (size_t i = 0; i < n; ++i) 
        B_prime[i] = args.b_features[i] * (1.0f - G[i]) * avg_a;

    // Stage 6: Context Integration 
    for (size_t i = 0; i < n; ++i) 
        C_prime[i] = args.c_features[i] + (Smooth_A[i] / (1.0f + std::abs(Smooth_A[i])));

    // Stage 7: Hidden Interaction
    for (size_t i = 0; i < n; ++i) 
        H[i] = Smooth_A[i] * C_prime[i];

    // Stage 8: Normalization
    for (size_t i = 0; i < n; ++i) 
        E[i] = (H[i] + B_prime[i]) / (1.0f + std::abs(Smooth_A[i]));

    // Stage 9: Final Output (ReLU)
    for (size_t i = 0; i < n; ++i) {
        float result = C_prime[i] - E[i];
        args.f_output[i] = std::max(result, 0.0f);
    }
}

// -------------------------------------------------------------------------
// TODO: Student Implementation
// -------------------------------------------------------------------------
void stu_grff(grff_args& args) {
    const size_t n = args.a_features.size();
    if (n == 0) return;

    std::vector<float> G(n), A_prime(n);

    const float* __restrict afeatptr = args.a_features.data();
    const float* __restrict bfeatptr = args.b_features.data();
    const float* __restrict cfeatptr = args.c_features.data();
    float* __restrict outptr = args.f_output.data();
    float* __restrict ap = A_prime.data();
    float* __restrict gptr = G.data();

    // Stage 1-3
    // Keep this scalar to preserve summation order for sum_a.
    float sum_a = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        const float prod = afeatptr[i] * bfeatptr[i];
        gptr[i] = 0.5f * (prod / (1.0f + std::abs(prod)) + 1.0f);
        ap[i] = afeatptr[i] + gptr[i];
        sum_a += ap[i];
    }
    const float avg_a = sum_a / static_cast<float>(n);

    // i = 0
    {
        const float smooth0 = ap[0];
        const float denom0 = 1.0f + std::abs(smooth0);
        const float invdenom0 = 1.0f / denom0;
        const float cprime0 = cfeatptr[0] + smooth0 * invdenom0;
        const float b0 = bfeatptr[0] * (1.0f - gptr[0]) * avg_a;
        const float h0 = smooth0 * cprime0;
        const float e0 = (h0 + b0) * invdenom0;
        outptr[0] = std::max(cprime0 - e0, 0.0f);
    }

    // Stage 4-9
    size_t i = 1;
    for (; i + 3 < n; i += 4) {
        const float apm1 = ap[i - 1];
        const float ap0  = ap[i];
        const float ap1  = ap[i + 1];
        const float ap2  = ap[i + 2];
        const float ap3  = ap[i + 3];

        const float smooth0 = (apm1 + ap0) * 0.5f;
        const float denom0 = 1.0f + std::abs(smooth0);
        const float invdenom0 = 1.0f / denom0;
        const float cprime0 = cfeatptr[i] + smooth0 * invdenom0;
        const float b0 = bfeatptr[i] * (1.0f - gptr[i]) * avg_a;
        const float h0 = smooth0 * cprime0;
        const float e0 = (h0 + b0) * invdenom0;
        outptr[i] = std::max(cprime0 - e0, 0.0f);

        const float smooth1 = (ap0 + ap1) * 0.5f;
        const float denom1 = 1.0f + std::abs(smooth1);
        const float invdenom1 = 1.0f / denom1;
        const float cprime1 = cfeatptr[i + 1] + smooth1 * invdenom1;
        const float b1 = bfeatptr[i + 1] * (1.0f - gptr[i + 1]) * avg_a;
        const float h1 = smooth1 * cprime1;
        const float e1 = (h1 + b1) * invdenom1;
        outptr[i + 1] = std::max(cprime1 - e1, 0.0f);

        const float smooth2 = (ap1 + ap2) * 0.5f;
        const float denom2 = 1.0f + std::abs(smooth2);
        const float invdenom2 = 1.0f / denom2;
        const float cprime2 = cfeatptr[i + 2] + smooth2 * invdenom2;
        const float b2 = bfeatptr[i + 2] * (1.0f - gptr[i + 2]) * avg_a;
        const float h2 = smooth2 * cprime2;
        const float e2 = (h2 + b2) * invdenom2;
        outptr[i + 2] = std::max(cprime2 - e2, 0.0f);

        const float smooth3 = (ap2 + ap3) * 0.5f;
        const float denom3 = 1.0f + std::abs(smooth3);
        const float invdenom3 = 1.0f / denom3;
        const float cprime3 = cfeatptr[i + 3] + smooth3 * invdenom3;
        const float b3 = bfeatptr[i + 3] * (1.0f - gptr[i + 3]) * avg_a;
        const float h3 = smooth3 * cprime3;
        const float e3 = (h3 + b3) * invdenom3;
        outptr[i + 3] = std::max(cprime3 - e3, 0.0f);
    }

    for (; i < n; ++i) {
        const float smooth0 = (ap[i - 1] + ap[i]) * 0.5f;
        const float denom0 = 1.0f + std::abs(smooth0);
        const float invdenom0 = 1.0f / denom0;
        const float cprime0 = cfeatptr[i] + smooth0 * invdenom0;
        const float b0 = bfeatptr[i] * (1.0f - gptr[i]) * avg_a;
        const float h0 = smooth0 * cprime0;
        const float e0 = (h0 + b0) * invdenom0;
        outptr[i] = std::max(cprime0 - e0, 0.0f);
    }
}

// -------------------------------------------------------------------------
// Wrappers and Checker
// -------------------------------------------------------------------------
void naive_grff_wrapper(void *ctx) {
    auto &args = *static_cast<grff_args *>(ctx);
    naive_grff(args);
}

void stu_grff_wrapper(void *ctx) {
    auto &args = *static_cast<grff_args *>(ctx);
    stu_grff(args);
}

bool grff_check(void *stu_ctx, void *ref_ctx, lab_test_func naive_func) {
    naive_func(ref_ctx);

    auto &stu_args = *static_cast<grff_args *>(stu_ctx);
    auto &ref_args = *static_cast<grff_args *>(ref_ctx);
    const auto eps = ref_args.epsilon;
    const double atol = 1e-6;

    if (stu_args.f_output.size() != ref_args.f_output.size()) return false;

    for (size_t i = 0; i < ref_args.f_output.size(); ++i) {
        double r = static_cast<double>(ref_args.f_output[i]);
        double s = static_cast<double>(stu_args.f_output[i]);
        double err = std::abs(s - r);

        if (err > (atol + eps * std::abs(r))) {
            debug_log("DEBUG: GRFF fail at %zu: ref=%f stu=%f\n", i, r, s);
            return false;
        }
    }
    return true;
}
