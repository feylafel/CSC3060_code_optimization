#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <print>
#include <vector>

#include "bench.h"
#include "matmul.h"

int main() {
    std::uint32_t seed = 12345u;
    constexpr int n = 512;

    matmul_args matmul_args_stu;
    matmul_args matmul_args_ref;
    initialize_matmul(matmul_args_stu, n, seed);
    initialize_matmul(matmul_args_ref, n, seed);

    std::println("\tMatMul: n={}", n);

    std::vector<bench_t> benchmarks = {
        {"MatMul (Student)",
         stu_matmul_wrapper,
         naive_matmul_wrapper,
         matmul_check,
         &matmul_args_stu,
         &matmul_args_ref,
         BASELINE_MATMUL},
    };

    std::cout << "\nRunning Benchmarks...\n";
    std::cout << "--------------------------------------------------------\n";
    std::cout << std::left << std::setw(25) << "Benchmark" << std::setw(12)
              << "Status" << std::right << std::setw(15) << "Nanoseconds"
              << "\n";
    std::cout << "--------------------------------------------------------\n";

    constexpr int k_best = 20;
    for (const auto& bench : benchmarks) {
        std::chrono::nanoseconds avg_time{0};

        for (int i = 0; i < k_best; ++i) {
            flush_cache();
            const auto elapsed = measure_time([&] { bench.tfunc(bench.args); });
            avg_time += elapsed;
        }
        avg_time /= static_cast<uint64_t>(k_best);

        const bool correct =
            bench.checkFunc(bench.args, bench.ref_args, bench.naiveFunc);

        std::cout << std::left << std::setw(25) << bench.description;
        if (!correct) {
            std::cout << "\033[1;31mFAILED\033[0m" << std::right
                      << std::setw(15) << "N/A" << "\n";
            std::cout
                << "  Error: Results do not match naive implementation!\n";
        } else {
            std::cout << "\033[1;32mPASSED\033[0m" << std::right
                      << std::setw(15) << avg_time.count() << " ns";
            if (avg_time.count() > bench.baseline_time.count() * 1.1) {
                std::cout << " (SLOW)";
            }
            std::cout << "\n";
        }
    }
    return 0;
}



/*
--------- BITWISE BENCH ------------

#include "bitwise.h"
int main() {
    std::uint32_t seed = 12345u;
    constexpr std::size_t bitwise_size = 1024000;
    bitwise_args bitwise_args_stu;
    bitwise_args bitwise_args_ref;
    initialize_bitwise(&bitwise_args_stu, bitwise_size, seed);
    initialize_bitwise(&bitwise_args_ref, bitwise_size, seed);
    std::println("\tBitwise: vector length={}", bitwise_size);
    std::vector<bench_t> benchmarks = {
        {"Bitwise (Student)",
         stu_bitwise_wrapper,
         naive_bitwise_wrapper,
         bitwise_check,
         &bitwise_args_stu,
         &bitwise_args_ref,
         BASELINE_BITWISE},
    };
    std::cout << "\nRunning Benchmarks...\n";
    std::cout << "--------------------------------------------------------\n";
    std::cout << std::left << std::setw(25) << "Benchmark" << std::setw(12)
              << "Status" << std::right << std::setw(15) << "Nanoseconds"
              << "\n";
    std::cout << "--------------------------------------------------------\n";
    constexpr int k_best = 20;
    for (const auto &bench : benchmarks) {
        std::chrono::nanoseconds avg_time{0};
        for (int i = 0; i < k_best; ++i) {
            flush_cache();
            const auto elapsed = measure_time([&] { bench.tfunc(bench.args); });
            avg_time += elapsed;
        }
        avg_time /= static_cast<uint64_t>(k_best);
        const bool correct =
            bench.checkFunc(bench.args, bench.ref_args, bench.naiveFunc);
        std::cout << std::left << std::setw(25) << bench.description;
        if (!correct) {
            std::cout << "\033[1;31mFAILED\033[0m" << std::right
                      << std::setw(15) << "N/A" << "\n";
            std::cout
                << "  Error: Results do not match naive implementation!\n";
        } else {
            std::cout << "\033[1;32mPASSED\033[0m" << std::right
                      << std::setw(15) << avg_time.count() << " ns";
            if (avg_time.count() > bench.baseline_time.count() * 1.1) {
                std::cout << " (SLOW)";
            }
            std::cout << "\n";
        }
    }
    return 0;
}
*/