#include "filter_gradient.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>

void convert_filter_gradient_data_to_grouped(filter_gradient_args* args) {
    if (!args) {
        return;
    }
    const std::size_t n = args->width * args->height;
    if (n == 0) {
        return;
    }
    const std::size_t t = 3 * n;
    data_struct& d = args->data;
    filter_gradient_grouped& g = args->grouped;

    g.abc.resize(t);
    g.def.resize(t);
    g.ghi.resize(t);

    for (std::size_t k = 0; k < n; ++k) {
        const std::size_t b = 3 * k;
        g.abc[b + 0] = d.a[k];
        g.abc[b + 1] = d.b[k];
        g.abc[b + 2] = d.c[k];
        g.def[b + 0] = d.d[k];
        g.def[b + 1] = d.e[k];
        g.def[b + 2] = d.f[k];
        g.ghi[b + 0] = d.g[k];
        g.ghi[b + 1] = d.h[k];
        g.ghi[b + 2] = d.i[k];
    }
}

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
    // Conversion is not part of the timed stu kernel (runs in init, beforeenchmarks).
    convert_filter_gradient_data_to_grouped(args);
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

void stu_filter_gradient(float& out, const filter_gradient_grouped& g,
                   std::size_t width, std::size_t height) {
    // TODO: You may need to add a function to convert data structure (not
    // included in time measurement), then implement your version in
    // stu_filter_gradient, whch is called by stu_filter_gradient_wrapper.
    const std::size_t W = width;
    const std::size_t H = height;
    constexpr float inv9 = 1.0f / 9.0f;

    const float* const abc = g.abc.data();
    const float* const def = g.def.data();
    const float* const ghi = g.ghi.data();

    double total = 0.0;

    for (std::size_t y = 1; y + 1 < H; ++y) {
        const std::size_t ym1 = y - 1;
        const std::size_t y0 = y;
        const std::size_t yp1 = y + 1;

        const std::size_t o_m1 = 3 * (ym1 * W);
        const std::size_t o_0 = 3 * (y0 * W);
        const std::size_t o_p1 = 3 * (yp1 * W);

        const float* abc_m1 = abc + o_m1;
        const float* abc_0 = abc + o_0;
        const float* abc_p1 = abc + o_p1;

        const float* def_m1 = def + o_m1;
        const float* def_0 = def + o_0;
        const float* def_p1 = def + o_p1;

        const float* ghi_m1 = ghi + o_m1;
        const float* ghi_p1 = ghi + o_p1;

        // 3x3 box on a,b,c: sliding x on grouped stream; a at 3*col+0, b +1, c +2.
        float sum_a = abc_m1[0] + abc_m1[3] + abc_m1[6] + abc_0[0] + abc_0[3] +
                      abc_0[6] + abc_p1[0] + abc_p1[3] + abc_p1[6];
        float sum_b = abc_m1[1] + abc_m1[4] + abc_m1[7] + abc_0[1] + abc_0[4] +
                      abc_0[7] + abc_p1[1] + abc_p1[4] + abc_p1[7];
        float sum_c = abc_m1[2] + abc_m1[5] + abc_m1[8] + abc_0[2] + abc_0[5] +
                      abc_0[8] + abc_p1[2] + abc_p1[5] + abc_p1[8];

        for (std::size_t x = 1; x + 1 < W; ++x) {
            if (x > 1) {
                const std::size_t c_out = x - 2;
                const std::size_t c_in = x + 1;
                const std::size_t o_out = 3 * c_out;
                const std::size_t o_in = 3 * c_in;
                sum_a += -abc_m1[o_out] - abc_0[o_out] - abc_p1[o_out] + abc_m1[o_in] +
                         abc_0[o_in] + abc_p1[o_in];
                sum_b += -abc_m1[o_out + 1] - abc_0[o_out + 1] - abc_p1[o_out + 1] +
                         abc_m1[o_in + 1] + abc_0[o_in + 1] + abc_p1[o_in + 1];
                sum_c += -abc_m1[o_out + 2] - abc_0[o_out + 2] - abc_p1[o_out + 2] +
                         abc_m1[o_in + 2] + abc_0[o_in + 2] + abc_p1[o_in + 2];
            }

            const std::size_t xm1 = x - 1;
            const std::size_t x0 = x;
            const std::size_t xp1 = x + 1;

            const std::size_t jm1 = 3 * xm1;
            const std::size_t j0 = 3 * x0;
            const std::size_t jp1 = 3 * xp1;

            const float avg_a = sum_a * inv9;
            const float avg_b = sum_b * inv9;
            const float avg_c = sum_c * inv9;
            const float p1 = avg_a * avg_b + avg_c;

            const float sobel_dx =
                -def_m1[jm1 + 0] + def_m1[jp1 + 0] - 2.0f * def_0[jm1 + 0] +
                2.0f * def_0[jp1 + 0] - def_p1[jm1 + 0] + def_p1[jp1 + 0];
            const float sobel_ex =
                -def_m1[jm1 + 1] + def_m1[jp1 + 1] - 2.0f * def_0[jm1 + 1] +
                2.0f * def_0[jp1 + 1] - def_p1[jm1 + 1] + def_p1[jp1 + 1];
            const float sobel_fx =
                -def_m1[jm1 + 2] + def_m1[jp1 + 2] - 2.0f * def_0[jm1 + 2] +
                2.0f * def_0[jp1 + 2] - def_p1[jm1 + 2] + def_p1[jp1 + 2];
            const float p2 = sobel_dx * sobel_ex + sobel_fx;

            const float sobel_gy = -ghi_m1[jm1 + 0] - 2.0f * ghi_m1[j0 + 0] -
                                   ghi_m1[jp1 + 0] + ghi_p1[jm1 + 0] +
                                   2.0f * ghi_p1[j0 + 0] + ghi_p1[jp1 + 0];
            const float sobel_hy = -ghi_m1[jm1 + 1] - 2.0f * ghi_m1[j0 + 1] -
                                   ghi_m1[jp1 + 1] + ghi_p1[jm1 + 1] +
                                   2.0f * ghi_p1[j0 + 1] + ghi_p1[jp1 + 1];
            const float sobel_iy = -ghi_m1[jm1 + 2] - 2.0f * ghi_m1[j0 + 2] -
                                   ghi_m1[jp1 + 2] + ghi_p1[jm1 + 2] +
                                   2.0f * ghi_p1[j0 + 2] + ghi_p1[jp1 + 2];
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
    stu_filter_gradient(args.out, args.grouped, args.width, args.height);
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
