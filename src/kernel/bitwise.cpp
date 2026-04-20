#include "bitwise.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>

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
    constexpr std::uint64_t kMaskLo8 = 0x5A5A5A5A5A5A5A5AULL;
    constexpr std::uint64_t kMaskHi8 = 0xC3C3C3C3C3C3C3C3ULL;
    const std::size_t n = std::min({result.size(), a.size(), b.size()});
    auto *const out = reinterpret_cast<unsigned char *>(result.data());
    const auto *const pa = reinterpret_cast<const unsigned char *>(a.data());
    const auto *const pb = reinterpret_cast<const unsigned char *>(b.data());
    std::size_t i = 0;
    const std::size_t n8 = n - (n % 8);
        for (; i < n8; i += 8) {
            std::uint64_t ua = 0;
            std::uint64_t ub = 0;
            std::memcpy(&ua, pa + i, 8);
            std::memcpy(&ub, pb + i, 8);
            const std::uint64_t shared = ua & ub;
            const std::uint64_t either = ua | ub;
            const std::uint64_t diff = ua ^ ub;
            const std::uint64_t mixed0 =
                (diff & kMaskLo8) | ((~shared) & (~kMaskLo8));
            const std::uint64_t mixed1 =
                ((either ^ kMaskHi8) & (shared | (~kMaskHi8))) ^ diff;
            const std::uint64_t packed = mixed0 ^ mixed1;
            std::memcpy(out + i, &packed, 8);
        }
        for (; i < n; ++i) {
            const auto ua = static_cast<std::uint8_t>(a[i]);
            const auto ub = static_cast<std::uint8_t>(b[i]);
            const auto shared = static_cast<std::uint8_t>(ua & ub);
            const auto either = static_cast<std::uint8_t>(ua | ub);
            const auto diff = static_cast<std::uint8_t>(ua ^ ub);
            const auto mixed0 = static_cast<std::uint8_t>(
                (diff & kMaskLo) | (~shared & ~kMaskLo));
            const auto mixed1 = static_cast<std::uint8_t>(
                ((either ^ kMaskHi) & (shared | ~kMaskHi)) ^ diff);
            result[i] = static_cast<std::int8_t>(mixed0 ^ mixed1);
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
