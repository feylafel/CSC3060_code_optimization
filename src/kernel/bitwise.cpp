#include "bitwise.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>

// #include <immintrin.h>

namespace {
    inline std::uint64_t bitwise_lane_u64(std::uint64_t va, std::uint64_t vb,
                                          std::uint64_t mlo64, std::uint64_t mhi64,
                                          std::uint64_t nlo64, std::uint64_t nhi64) {
        const std::uint64_t s = va & vb;
        const std::uint64_t e = va | vb;
        const std::uint64_t d = va ^ vb;
        constexpr std::uint64_t ALL = ~std::uint64_t{0};
        const std::uint64_t mixed0 = (d & mlo64) | ((s ^ ALL) & nlo64);
        const std::uint64_t mixed1 = (((e ^ mhi64) & (s | nhi64)) ^ d);
        return mixed0 ^ mixed1;
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

// TODO: Optimize the bitwise function
void stu_bitwise(std::span<std::int8_t> result, std::span<const std::int8_t> a,
                 std::span<const std::int8_t> b) {
                 constexpr std::uint8_t kMaskLo = 0x5Au;
                 constexpr std::uint8_t kMaskHi = 0xC3u;
                 constexpr std::uint64_t mlo64 = 0x5A5A5A5A5A5A5A5AULL;
                 constexpr std::uint64_t mhi64 = 0xC3C3C3C3C3C3C3C3ULL;
                 constexpr std::uint64_t nlo64 = 0xA5A5A5A5A5A5A5A5ULL;
                 constexpr std::uint64_t nhi64 = 0x3C3C3C3C3C3C3C3CULL;
                 const std::size_t n = std::min({result.size(), a.size(), b.size()});
                 auto *out = result.data();
                 const auto *pa = a.data();
                 const auto *pb = b.data();
                 std::size_t i = 0;
                 const std::size_t n8 = n & ~std::size_t{7};
                 for (; i < n8; i += 8) {
                     std::uint64_t va = 0;
                     std::uint64_t vb = 0;
                     std::memcpy(&va, pa + i, sizeof(va));
                     std::memcpy(&vb, pb + i, sizeof(vb));
                     const std::uint64_t packed =
                         bitwise_lane_u64(va, vb, mlo64, mhi64, nlo64, nhi64);
                     std::memcpy(out + i, &packed, sizeof(packed));
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
