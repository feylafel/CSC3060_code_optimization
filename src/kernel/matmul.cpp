#include "matmul.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

#include <iostream>

void initialize_matmul(matmul_args& args, int n, uint32_t seed) {
    if (n <= 0) {
        throw std::invalid_argument("initialize_matmul: n must be positive.");
    }

    args.n = n;
    args.epsilon = 1e-3;

    const size_t elem_count = static_cast<size_t>(n) * static_cast<size_t>(n);
    args.A.resize(elem_count);
    args.B.resize(elem_count);
    args.C.assign(elem_count, 0.0f);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < elem_count; ++i) {
        args.A[i] = dist(rng);
        args.B[i] = dist(rng);
    }
}

void naive_matmul(std::vector<float>& C,
                  const std::vector<float>& A,
                  const std::vector<float>& B,
                  int n) {
    std::fill(C.begin(), C.end(), 0.0f);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

void stu_matmul(std::vector<float>& C,
                const std::vector<float>& A,
                const std::vector<float>& B,
                int n) {
    std::fill(C.begin(), C.end(), 0.0f);
    const int blocksz = 32;
    const int jblocksz = 64;
    for (int iblock = 0; iblock < n; iblock += blocksz)
    {
        int ilim = std::min(iblock + blocksz, n);
        for (int kblock = 0; kblock < n; kblock += blocksz)
        {
            int klim = std::min(kblock + blocksz, n);
            for (int jblock = 0; jblock < n; jblock += jblocksz)
            {
                int jlim = std::min(jblock + jblocksz, n);
                for (int i = iblock; i < ilim; ++i)
                {
                    float* cptr = &C[i * n];
                    for (int k = kblock; k < klim; ++k)
                    {
                        float aval = A[i * n + k];
                        const float* bptr = &B[k * n];
                        int j = jblock;

                        for (; j + 7 < jlim; j += 8)
                        {
                            *(cptr + j) += *(bptr + j) * aval;
                            *(cptr + j + 1) += *(bptr + j + 1) * aval;
                            *(cptr + j + 2) += *(bptr + j + 2) * aval;
                            *(cptr + j + 3) += *(bptr + j + 3) * aval;
                            *(cptr + j + 4) += *(bptr + j + 4) * aval;
                            *(cptr + j + 5) += *(bptr + j + 5) * aval;
                            *(cptr + j + 6) += *(bptr + j + 6) * aval;
                            *(cptr + j + 7) += *(bptr + j + 7) * aval;
                        }
                        for (; j < jlim; ++j)
                        {
                            *(cptr + j) += *(bptr + j) * aval;
                        }
                    }
                }
            }
        }
    }
}


void naive_matmul_wrapper(void* ctx) {
    auto& args = *static_cast<matmul_args*>(ctx);
    naive_matmul(args.C, args.A, args.B, args.n);
}

void stu_matmul_wrapper(void* ctx) {
    auto& args = *static_cast<matmul_args*>(ctx);
    stu_matmul(args.C, args.A, args.B, args.n);
}

bool matmul_check(void* stu_ctx, void* ref_ctx, lab_test_func naive_func) {
    naive_func(ref_ctx);

    auto& stu_args = *static_cast<matmul_args*>(stu_ctx);
    auto& ref_args = *static_cast<matmul_args*>(ref_ctx);

    if (stu_args.C.size() != ref_args.C.size()) {
        debug_log("\tDEBUG: matmul size mismatch: stu={} ref={}\n",
                  stu_args.C.size(),
                  ref_args.C.size());
        
        return false;
    }

    const double eps = ref_args.epsilon;
    const int n = ref_args.n;
    double max_rel = 0.0;
    size_t worst_idx = 0;

    for (size_t i = 0; i < ref_args.C.size(); ++i) {
        const double r = static_cast<double>(ref_args.C[i]);
        const double s = static_cast<double>(stu_args.C[i]);
        const double diff = std::abs(s - r);
        const double rel = (std::abs(r) > 1e-9) ? diff / std::abs(r) : diff;

        if (rel > max_rel) {
            max_rel = rel;
            worst_idx = i;
        }

        if (rel > eps) {
            const size_t row = (n > 0) ? (i / static_cast<size_t>(n)) : 0;
            const size_t col = (n > 0) ? (i % static_cast<size_t>(n)) : 0;
            debug_log("\tDEBUG: matmul fail at index {} (row={}, col={}): ref={} stu={} rel={} eps={}\n",
                      i,
                      row,
                      col,
                      ref_args.C[i],
                      stu_args.C[i],
                      rel,
                      eps);
            return false;
        }
    }

    debug_log("\tDEBUG: matmul_check passed. max_rel={} at index {}\n",
              max_rel,
              worst_idx);
    return true;
}
