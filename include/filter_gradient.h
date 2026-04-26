#ifndef FILTER_GRADIENT_H
#define FILTER_GRADIENT_H

#include "bench.h"
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

inline constexpr std::chrono::nanoseconds BASELINE_FILTER_GRADIENT{25000000};

struct data_struct {
    std::vector<float> a;
    std::vector<float> b;
    std::vector<float> c;
    std::vector<float> d;
    std::vector<float> e;
    std::vector<float> f;
    std::vector<float> g;
    std::vector<float> h;
    std::vector<float> i;
};

// Grouped SoA (v3): three streams with (a,b,c) | (d,e,f) | (g,h,i) interleaved
// per pixel in row-major order. For linear pixel index k: channel triple at
// [3k+0],[3k+1],[3k+2]. alignas(64) aligns the struct; heap buffers use
// 16+ byte alignment in practice for float vectors on this platform, which
// helps cache-line and SIMD-friendly 12-byte groups.
struct alignas(64) filter_gradient_grouped {
    std::vector<float> abc;
    std::vector<float> def;
    std::vector<float> ghi;
};

struct filter_gradient_args {
    data_struct data; 
    // TODO: You may want to add new params at the end...


    std::size_t width;
    std::size_t height;
    float out;
    double epsilon;
    filter_gradient_grouped grouped;

    explicit filter_gradient_args(double epsilon_in = 1e-6)
        : width(0), height(0), out(0.0f), epsilon(epsilon_in) {}
};

// TODO: You may need to add a function to convert data structure (not 
// included in time measurement), then implement your version in 
// stu_filter_gradient, whch is called by stu_filter_gradient_wrapper.
void convert_filter_gradient_data_to_grouped(filter_gradient_args* args);

void naive_filter_gradient(float& out, const data_struct& data,
                   std::size_t width, std::size_t height);
void stu_filter_gradient(float& out, const filter_gradient_grouped& g,
                   std::size_t width, std::size_t height);

void naive_filter_gradient_wrapper(void* ctx);
void stu_filter_gradient_wrapper(void* ctx);

void initialize_filter_gradient(filter_gradient_args* args,
                        std::size_t width,
                        std::size_t height,
                        std::uint_fast64_t seed);

bool filter_gradient_check(void* stu_ctx, void* ref_ctx, lab_test_func naive_func);

#endif // filter_gradient_H