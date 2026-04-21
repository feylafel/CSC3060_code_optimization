#include "bitwise.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>

#include <immintrin.h>

// Byte-parallel match to naive_bitwise: shared/either/diff then mixed0/mixed1.
// mixed0 second term: (~shared & ~kMaskLo) -> andnot(shared, not_mlo), not_mlo = mlo^0xFF.
namespace {

#if defined(__AVX2__)
inline __m256i bitwise_lane_avx2(__m256i va, __m256i vb, __m256i mlo, __m256i mhi,
                                 __m256i not_mlo, __m256i not_mhi) {
    const __m256i shared = _mm256_and_si256(va, vb);
    const __m256i either = _mm256_or_si256(va, vb);
    const __m256i diff = _mm256_xor_si256(va, vb);
    const __m256i mixed0 = _mm256_or_si256(
        _mm256_and_si256(diff, mlo), _mm256_andnot_si256(shared, not_mlo));
    const __m256i mixed1 = _mm256_xor_si256(
        _mm256_and_si256(_mm256_xor_si256(either, mhi),
                         _mm256_or_si256(shared, not_mhi)),
        diff);
    return _mm256_xor_si256(mixed0, mixed1);
}
#endif

inline __m128i bitwise_lane_sse(__m128i va, __m128i vb, __m128i mlo, __m128i mhi,
                                __m128i not_mlo, __m128i not_mhi) {
    const __m128i shared = _mm_and_si128(va, vb);
    const __m128i either = _mm_or_si128(va, vb);
    const __m128i diff = _mm_xor_si128(va, vb);
    const __m128i mixed0 = _mm_or_si128(_mm_and_si128(diff, mlo),
                                        _mm_andnot_si128(shared, not_mlo));
    const __m128i mixed1 =
        _mm_xor_si128(_mm_and_si128(_mm_xor_si128(either, mhi),
                                    _mm_or_si128(shared, not_mhi)),
                      diff);
    return _mm_xor_si128(mixed0, mixed1);
}

} // namespace

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

void stu_bitwise(std::span<std::int8_t> result, std::span<const std::int8_t> a,
                 std::span<const std::int8_t> b) {
    constexpr std::uint8_t kMaskLo = 0x5Au;
    constexpr std::uint8_t kMaskHi = 0xC3u;
    const std::size_t n = std::min({result.size(), a.size(), b.size()});
    auto *out = result.data();
    const auto *pa = a.data();
    const auto *pb = b.data();
    std::size_t i = 0;

#if defined(__AVX2__)
    const __m256i mlo256 = _mm256_set1_epi8(static_cast<char>(kMaskLo));
    const __m256i mhi256 = _mm256_set1_epi8(static_cast<char>(kMaskHi));
    const __m256i ones256 = _mm256_set1_epi8(static_cast<char>(0xFF));
    const __m256i not_mlo256 = _mm256_xor_si256(mlo256, ones256);
    const __m256i not_mhi256 = _mm256_xor_si256(mhi256, ones256);

    const std::size_t n64 = n & ~std::size_t{63};
    for (; i < n64; i += 64) {
        const __m256i va0 =
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(pa + i));
        const __m256i vb0 =
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(pb + i));
        const __m256i va1 =
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(pa + i + 32));
        const __m256i vb1 =
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(pb + i + 32));
        const __m256i o0 =
            bitwise_lane_avx2(va0, vb0, mlo256, mhi256, not_mlo256, not_mhi256);
        const __m256i o1 =
            bitwise_lane_avx2(va1, vb1, mlo256, mhi256, not_mlo256, not_mhi256);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(out + i), o0);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(out + i + 32), o1);
    }
    const std::size_t n32 = n & ~std::size_t{31};
    for (; i < n32; i += 32) {
        const __m256i va =
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(pa + i));
        const __m256i vb =
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(pb + i));
        const __m256i o =
            bitwise_lane_avx2(va, vb, mlo256, mhi256, not_mlo256, not_mhi256);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(out + i), o);
    }
#endif

    const __m128i mlo128 = _mm_set1_epi8(static_cast<char>(kMaskLo));
    const __m128i mhi128 = _mm_set1_epi8(static_cast<char>(kMaskHi));
    const __m128i ones128 = _mm_set1_epi8(static_cast<char>(0xFF));
    const __m128i not_mlo128 = _mm_xor_si128(mlo128, ones128);
    const __m128i not_mhi128 = _mm_xor_si128(mhi128, ones128);
    const std::size_t n16 = n & ~std::size_t{15};
    for (; i < n16; i += 16) {
        const __m128i va =
            _mm_loadu_si128(reinterpret_cast<const __m128i *>(pa + i));
        const __m128i vb =
            _mm_loadu_si128(reinterpret_cast<const __m128i *>(pb + i));
        const __m128i o =
            bitwise_lane_sse(va, vb, mlo128, mhi128, not_mlo128, not_mhi128);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), o);
    }
    for (; i < n; ++i) {
        const auto ua = static_cast<std::uint8_t>(pa[i]);
        const auto ub = static_cast<std::uint8_t>(pb[i]);
        const auto shared = static_cast<std::uint8_t>(ua & ub);
        const auto either = static_cast<std::uint8_t>(ua | ub);
        const auto diff = static_cast<std::uint8_t>(ua ^ ub);
        const auto mixed0 =
            static_cast<std::uint8_t>((diff & kMaskLo) | (~shared & ~kMaskLo));
        const auto mixed1 = static_cast<std::uint8_t>(
            ((either ^ kMaskHi) & (shared | ~kMaskHi)) ^ diff);
        out[i] = static_cast<std::int8_t>(mixed0 ^ mixed1);
    }
}

void naive_bitwise_wrapper(void *ctx) {
    auto &args = *static_cast<bitwise_args *>(ctx);
    naive_bitwise(args.result, args.a, args.b);
}

void stu_bitwise_wrapper(void *ctx) {
    auto &args = *static_cast<bitwise_args *>(ctx);
    stu_bitwise(args.result, args.a, args.b);
}

bool bitwise_check(void *stu_ctx, void *ref_ctx, lab_test_func naive_func) {
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