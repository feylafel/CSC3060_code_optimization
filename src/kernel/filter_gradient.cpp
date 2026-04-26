#include "filter_gradient.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>

void initialize_filter_gradient(filter_gradient_args* args,
                        std::size_t width,
                        std::size_t height,
                        std::uint_fast64_t seed) {
    if (!args) {
        return;
    }

    assert(width >= 3);
    assert(height >= 3);

    args->width = width;
    args->height = height;
    args->out = 0.0f;

    const std::size_t count = width * height;

    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    args->data.a.resize(count);
    args->data.b.resize(count);
    args->data.c.resize(count);
    args->data.d.resize(count);
    args->data.e.resize(count);
    args->data.f.resize(count);
    args->data.g.resize(count);
    args->data.h.resize(count);
    args->data.i.resize(count);

    for (std::size_t k = 0; k < count; ++k) {
        args->data.a[k] = dist(gen);
        args->data.b[k] = dist(gen);
        args->data.c[k] = dist(gen);
        args->data.d[k] = dist(gen);
        args->data.e[k] = dist(gen);
        args->data.f[k] = dist(gen);
        args->data.g[k] = dist(gen);
        args->data.h[k] = dist(gen);
        args->data.i[k] = dist(gen);
    }
    build_filter_gradient_opt_ctx(args);
}
  
void build_filter_gradient_opt_ctx(filter_gradient_args* args) {
    if (!args) return;
    args->opt.a = args->data.a.data(); args->opt.b = args->data.b.data(); args->opt.c = args->data.c.data();
    args->opt.d = args->data.d.data(); args->opt.e = args->data.e.data(); args->opt.f = args->data.f.data();
    args->opt.g = args->data.g.data(); args->opt.h = args->data.h.data(); args->opt.i = args->data.i.data();
    args->opt.width = args->width;
    args->opt.height = args->height;
}

void naive_filter_gradient(float& out, const data_struct& data,
                   std::size_t width, std::size_t height) {
    const std::size_t W = width;
    const std::size_t H = height;
    constexpr float inv9 = 1.0f / 9.0f;

    double total = 0.0f;

    for (std::size_t y = 1; y + 1 < H; ++y) {
        for (std::size_t x = 1; x + 1 < W; ++x) {

            double sum_a = 0.0, sum_b = 0.0, sum_c = 0.0;
            for (int dy = -1; dy <= 1; ++dy) {
                const std::size_t row = (y + dy) * W;
                for (int dx = -1; dx <= 1; ++dx) {
                    const std::size_t idx = row + (x + dx);
                    sum_a += data.a[idx];
                    sum_b += data.b[idx];
                    sum_c += data.c[idx];
                }
            }
            const float avg_a = sum_a * inv9;
            const float avg_b = sum_b * inv9;
            const float avg_c = sum_c * inv9;
            const float p1 = avg_a * avg_b + avg_c;

            const std::size_t ym1 = (y - 1) * W;
            const std::size_t y0  = y * W;
            const std::size_t yp1 = (y + 1) * W;

            const std::size_t xm1 = x - 1;
            const std::size_t x0  = x;
            const std::size_t xp1 = x + 1;

            const float sobel_dx =
                -data.d[ym1 + xm1] + data.d[ym1 + xp1]
                -2.0f * data.d[y0 + xm1] + 2.0f * data.d[y0 + xp1]
                -data.d[yp1 + xm1] + data.d[yp1 + xp1];

            const float sobel_ex =
                -data.e[ym1 + xm1] + data.e[ym1 + xp1]
                -2.0f * data.e[y0 + xm1] + 2.0f * data.e[y0 + xp1]
                -data.e[yp1 + xm1] + data.e[yp1 + xp1];

            const float sobel_fx =
                -data.f[ym1 + xm1] + data.f[ym1 + xp1]
                -2.0f * data.f[y0 + xm1] + 2.0f * data.f[y0 + xp1]
                -data.f[yp1 + xm1] + data.f[yp1 + xp1];

            const float p2 = sobel_dx * sobel_ex + sobel_fx;

            const float sobel_gy =
                -data.g[ym1 + xm1] - 2.0f * data.g[ym1 + x0] - data.g[ym1 + xp1]
                + data.g[yp1 + xm1] + 2.0f * data.g[yp1 + x0] + data.g[yp1 + xp1];

            const float sobel_hy =
                -data.h[ym1 + xm1] - 2.0f * data.h[ym1 + x0] - data.h[ym1 + xp1]
                + data.h[yp1 + xm1] + 2.0f * data.h[yp1 + x0] + data.h[yp1 + xp1];

            const float sobel_iy =
                -data.i[ym1 + xm1] - 2.0f * data.i[ym1 + x0] - data.i[ym1 + xp1]
                + data.i[yp1 + xm1] + 2.0f * data.i[yp1 + x0] + data.i[yp1 + xp1];

            const float p3 = sobel_gy * sobel_hy + sobel_iy;

            total += p1 + p2 + p3;
        }
    }

    out = total;
}

void stu_filter_gradient(float& out, const data_struct& data,
                   std::size_t width, std::size_t height) {
    // TODO: You may need to add a function to convert data structure (not 
    // included in time measurement), then implement your version in 
    // stu_filter_gradient, whch is called by stu_filter_gradient_wrapper.
    const std::size_t W = width;
    const std::size_t H = height;
    constexpr float inv9 = 1.0f / 9.0f;

    double total = 0.0;

    for (std::size_t y = 1; y + 1 < H; ++y) {
        const std::size_t ym1 = y - 1;
        const std::size_t y0 = y;
        const std::size_t yp1 = y + 1;

        const float* a_m1 = data.a.data() + ym1 * W;
        const float* a_0 = data.a.data() + y0 * W;
        const float* a_p1 = data.a.data() + yp1 * W;
        const float* b_m1 = data.b.data() + ym1 * W;
        const float* b_0 = data.b.data() + y0 * W;
        const float* b_p1 = data.b.data() + yp1 * W;
        const float* c_m1 = data.c.data() + ym1 * W;
        const float* c_0 = data.c.data() + y0 * W;
        const float* c_p1 = data.c.data() + yp1 * W;

        const float* d_m1 = data.d.data() + ym1 * W;
        const float* d_0 = data.d.data() + y0 * W;
        const float* d_p1 = data.d.data() + yp1 * W;
        const float* e_m1 = data.e.data() + ym1 * W;
        const float* e_0 = data.e.data() + y0 * W;
        const float* e_p1 = data.e.data() + yp1 * W;
        const float* f_m1 = data.f.data() + ym1 * W;
        const float* f_0 = data.f.data() + y0 * W;
        const float* f_p1 = data.f.data() + yp1 * W;

        const float* g_m1 = data.g.data() + ym1 * W;
        const float* g_0 = data.g.data() + y0 * W;
        const float* g_p1 = data.g.data() + yp1 * W;
        const float* h_m1 = data.h.data() + ym1 * W;
        const float* h_0 = data.h.data() + y0 * W;
        const float* h_p1 = data.h.data() + yp1 * W;
        const float* i_m1 = data.i.data() + ym1 * W;
        const float* i_0 = data.i.data() + y0 * W;
        const float* i_p1 = data.i.data() + yp1 * W;

        for (std::size_t x = 1; x + 1 < W; ++x) {
            const std::size_t xm1 = x - 1;
            const std::size_t x0 = x;
            const std::size_t xp1 = x + 1;

            const float sum_a = a_m1[xm1] + a_m1[x0] + a_m1[xp1] +
                                a_0[xm1] + a_0[x0] + a_0[xp1] +
                                a_p1[xm1] + a_p1[x0] + a_p1[xp1];
            const float sum_b = b_m1[xm1] + b_m1[x0] + b_m1[xp1] +
                                b_0[xm1] + b_0[x0] + b_0[xp1] +
                                b_p1[xm1] + b_p1[x0] + b_p1[xp1];
            const float sum_c = c_m1[xm1] + c_m1[x0] + c_m1[xp1] +
                                c_0[xm1] + c_0[x0] + c_0[xp1] +
                                c_p1[xm1] + c_p1[x0] + c_p1[xp1];

            const float avg_a = sum_a * inv9;
            const float avg_b = sum_b * inv9;
            const float avg_c = sum_c * inv9;
            const float p1 = avg_a * avg_b + avg_c;

            const float sobel_dx = -d_m1[xm1] + d_m1[xp1] -
                                   2.0f * d_0[xm1] + 2.0f * d_0[xp1] -
                                   d_p1[xm1] + d_p1[xp1];
            const float sobel_ex = -e_m1[xm1] + e_m1[xp1] -
                                   2.0f * e_0[xm1] + 2.0f * e_0[xp1] -
                                   e_p1[xm1] + e_p1[xp1];
            const float sobel_fx = -f_m1[xm1] + f_m1[xp1] -
                                   2.0f * f_0[xm1] + 2.0f * f_0[xp1] -
                                   f_p1[xm1] + f_p1[xp1];
            const float p2 = sobel_dx * sobel_ex + sobel_fx;

            const float sobel_gy = -g_m1[xm1] - 2.0f * g_m1[x0] - g_m1[xp1] +
                                   g_p1[xm1] + 2.0f * g_p1[x0] + g_p1[xp1];
            const float sobel_hy = -h_m1[xm1] - 2.0f * h_m1[x0] - h_m1[xp1] +
                                   h_p1[xm1] + 2.0f * h_p1[x0] + h_p1[xp1];
            const float sobel_iy = -i_m1[xm1] - 2.0f * i_m1[x0] - i_m1[xp1] +
                                   i_p1[xm1] + 2.0f * i_p1[x0] + i_p1[xp1];
            const float p3 = sobel_gy * sobel_hy + sobel_iy;

            total += static_cast<double>(p1 + p2 + p3);
        }
    }

    out = static_cast<float>(total);
}

void naive_filter_gradient_wrapper(void* ctx) {
    auto& args = *static_cast<filter_gradient_args*>(ctx);
    args.out = 0.0f;
    naive_filter_gradient(args.out, args.data, args.width, args.height);
}
void stu_filter_gradient_wrapper(void* ctx) {
    auto& args = *static_cast<filter_gradient_args*>(ctx);
    args.out = 0.0f;
    stu_filter_gradient(args.out, args.data, args.width, args.height);
}

bool filter_gradient_check(void* stu_ctx, void* ref_ctx, lab_test_func naive_func) {
    auto& stu_args = *static_cast<filter_gradient_args*>(stu_ctx);
    auto& ref_args = *static_cast<filter_gradient_args*>(ref_ctx);

    ref_args.out = 0.0f;
    naive_func(ref_ctx);

    const auto eps = ref_args.epsilon;
    const double s = static_cast<double>(stu_args.out);
    const double r = static_cast<double>(ref_args.out);
    const double err = std::abs(s - r);
    const double atol = 1e-6;
    const double rel = (std::abs(r) > atol) ? err / std::abs(r) : err;
    debug_log("DEBUG: filter_gradient stu={} ref={} err={} rel={}\n",
              stu_args.out,
              ref_args.out,
              err,
              rel);

    return err <= (atol + eps * std::abs(r));
}
