#include "bitwise.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <immintrin.h>

void initialize_bitwise(bitwise_args *args, const size_t size,
                        const std::uint_fast64_t seed) {
    if (!args) {
        return;
    }

    constexpr std::int8_t LOWER_BOUND = std::numeric_limits<std::int8_t>::min();
    constexpr std::int8_t UPPER_BOUND = std::numeric_limits<std::int8_t>::max();

    std::mt19937_64 gen(seed);
    std::uniform_int_distribution<int> dist(LOWER_BOUND, UPPER_BOUND);

    args->a.resize(size);
    args->b.resize(size);
    args->result.resize(size);

    for (std::size_t i = 0; i < size; ++i) {
        args->a[i] = static_cast<std::int8_t>(dist(gen));
        args->b[i] = static_cast<std::int8_t>(dist(gen));
        args->result[i] = 0;
    }
}

// The reference implementation of bitwise
// Student should not change this function
void naive_bitwise(std::span<std::int8_t> result,
                   std::span<const std::int8_t> a,
                   std::span<const std::int8_t> b) {
    constexpr std::uint8_t kMaskLo = 0x5Au;
    constexpr std::uint8_t kMaskHi = 0xC3u;

    const std::size_t n = std::min({result.size(), a.size(), b.size()});
    for (std::size_t i = 0; i < n; ++i) {
        const auto ua = static_cast<std::uint8_t>(a[i]);
        const auto ub = static_cast<std::uint8_t>(b[i]);

        const auto shared = static_cast<std::uint8_t>(ua & ub);
        const auto either = static_cast<std::uint8_t>(ua | ub);
        const auto diff = static_cast<std::uint8_t>(ua ^ ub);
        const auto mixed0 =
            static_cast<std::uint8_t>((diff & kMaskLo) | (~shared & ~kMaskLo));
        const auto mixed1 = static_cast<std::uint8_t>(
            ((either ^ kMaskHi) & (shared | ~kMaskHi)) ^ diff);

        result[i] = static_cast<std::int8_t>(mixed0 ^ mixed1);
    }
}

// Optimized: per-byte, same mapping as naive_bitwise, expressed as
//   out = 0xA5 ^ ((ua | ub) & 0x99)  with ua,ub taken as uint8_t bits.
// Vectorized: OR / AND / XOR on __m128i with byte-broadcast constants.
void stu_bitwise(std::span<std::int8_t> result, std::span<const std::int8_t> a,
                 std::span<const std::int8_t> b) {
    constexpr std::uint8_t kXorConst = 0xA5u;
    constexpr std::uint8_t kAndMask = 0x99u;

    const std::size_t n = std::min({result.size(), a.size(), b.size()});
    auto *out = result.data();
    const auto *pa = a.data();
    const auto *pb = b.data();

    const __m128i cA5 = _mm_set1_epi8(static_cast<char>(kXorConst));
    const __m128i c99 = _mm_set1_epi8(static_cast<char>(kAndMask));

    std::size_t i = 0;
    const std::size_t n32 = n & ~std::size_t{31};

    for (; i < n32; i += 32) {
        __m128i va0 =
            _mm_loadu_si128(reinterpret_cast<const __m128i *>(pa + i));
        __m128i vb0 =
            _mm_loadu_si128(reinterpret_cast<const __m128i *>(pb + i));
        __m128i va1 =
            _mm_loadu_si128(reinterpret_cast<const __m128i *>(pa + i + 16));
        __m128i vb1 =
            _mm_loadu_si128(reinterpret_cast<const __m128i *>(pb + i + 16));

        const __m128i either0 = _mm_or_si128(va0, vb0);
        const __m128i out0 = _mm_xor_si128(
            cA5, _mm_and_si128(either0, c99));
        const __m128i either1 = _mm_or_si128(va1, vb1);
        const __m128i out1 = _mm_xor_si128(
            cA5, _mm_and_si128(either1, c99));

        _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), out0);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i + 16), out1);
    }

    const std::size_t n16 = n & ~std::size_t{15};
    for (; i < n16; i += 16) {
        const __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pa + i));
        const __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pb + i));
        const __m128i either = _mm_or_si128(va, vb);
        const __m128i vr = _mm_xor_si128(cA5, _mm_and_si128(either, c99));
        _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), vr);
    }

    for (; i < n; ++i) {
        const auto ua = static_cast<std::uint8_t>(pa[i]);
        const auto ub = static_cast<std::uint8_t>(pb[i]);
        out[i] = static_cast<std::int8_t>(
            kXorConst ^ static_cast<std::uint8_t>((ua | ub) & kAndMask));
    }
}

void naive_bitwise_wrapper(void *ctx) {
    auto &args = *static_cast<bitwise_args *>(ctx);
    naive_bitwise(args.result, args.a, args.b);
}

void stu_bitwise_wrapper(void *ctx) {
    // Call your verion here
    auto &args = *static_cast<bitwise_args *>(ctx);
    stu_bitwise(args.result, args.a, args.b);
}

bool bitwise_check(void *stu_ctx, void *ref_ctx, lab_test_func naive_func) {
    // Compute reference
    naive_func(ref_ctx);

    auto &stu_args = *static_cast<bitwise_args *>(stu_ctx);
    auto &ref_args = *static_cast<bitwise_args *>(ref_ctx);

    if (stu_args.result.size() != ref_args.result.size()) {
        debug_log("\tDEBUG: size mismatch: stu={} ref={}\n",
                  stu_args.result.size(),
                  ref_args.result.size());
        return false;
    }

    std::int32_t max_abs_diff = 0;
    size_t worst_i = 0;

    for (size_t i = 0; i < ref_args.result.size(); ++i) {
        const auto r = static_cast<std::int32_t>(ref_args.result[i]);
        const auto s = static_cast<std::int32_t>(stu_args.result[i]);

        if (r != s) {
            max_abs_diff = std::abs(r - s);
            worst_i = i;

            debug_log("\tDEBUG: fail at {}: ref={} stu={} abs_diff={}\n",
                      i,
                      r,
                      s,
                      max_abs_diff);
            return false;
        }
    }

    debug_log("\tDEBUG: bitwise_check passed. max_abs_diff={} at i={}\n",
              max_abs_diff,
              worst_i);
    return true;
}