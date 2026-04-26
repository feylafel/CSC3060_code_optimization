#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <print>
#include <vector>

#include "bench.h"
#include "bitwise.h"
#include "blackscholes.h"
#include "filter_gradient.h"
#include "graph.h"
#include "grff.h"
#include "image_proc.h"
#include "matmul.h"
#include "relu.h"
#include "sparse_spmm.h"
#include "trace_replay.h"

enum class KernelTarget {
    BlackScholes,
    SparseSpMM,
    ReLU,
    Bitwise,
    MatMul,
    TraceReplay,
    Graph,
    GRFF,
    ImageProc,
    FilterGradient
};

int main() {
    std::uint32_t seed = 12345u;

    // ============================
    // Just change this line to the target kernel you want to benchmark
    constexpr KernelTarget kTarget = KernelTarget::FilterGradient;
    // ============================

    std::vector<bench_t> benchmarks;

    switch (kTarget) {
        case KernelTarget::BlackScholes: {
            blackscholes_args args_stu, args_ref;
            initialize_blackscholes(args_stu, 81920, seed);
            initialize_blackscholes(args_ref, 81920, seed);
            std::println("\tBlack-Scholes options: {}", args_stu.spot_price.size());

            benchmarks.push_back({"Black-Scholes (Student)",
                                  stu_BlkSchls_wrapper,
                                  naive_BlkSchls_wrapper,
                                  BlkSchls_check,
                                  &args_stu, &args_ref, BASELINE_BLACKSCHOLES});
            break;
        }

        case KernelTarget::SparseSpMM: {
            sparse_spmm_args args_stu, args_ref;
            initialize_spmm(args_stu, 512, 512, -1, {}, seed);
            initialize_spmm(args_ref, 512, 512, -1, {}, seed);
            std::println("\tSparse A (CSR): {} x {}, nnz={}",
                         args_stu.csr.rows,
                         args_stu.csr.cols,
                         args_stu.csr.values.size());

            benchmarks.push_back({"Sparse SpMM (Student)",
                                  stu_sparse_spmm_wrapper,
                                  naive_sparse_spmm_wrapper,
                                  sparse_spmm_check,
                                  &args_stu, &args_ref, BASELINE_SPARSE_SPMM});
            break;
        }

        case KernelTarget::ReLU: {
            constexpr std::size_t relu_size = 1024000;
            relu_args args_stu, args_ref;
            initialize_relu(&args_stu, relu_size, seed);
            initialize_relu(&args_ref, relu_size, seed);
            std::println("\tReLU: vector length={}", relu_size);

            benchmarks.push_back({"ReLU (Student)",
                                  stu_relu_wrapper,
                                  naive_relu_wrapper,
                                  relu_check,
                                  &args_stu, &args_ref, BASELINE_RELU});
            break;
        }

        case KernelTarget::Bitwise: {
            constexpr std::size_t bitwise_size = 1024000;
            bitwise_args args_stu, args_ref;
            initialize_bitwise(&args_stu, bitwise_size, seed);
            initialize_bitwise(&args_ref, bitwise_size, seed);
            std::println("\tBitwise: vector length={}", bitwise_size);

            benchmarks.push_back({"Bitwise (Student)",
                                  stu_bitwise_wrapper,
                                  naive_bitwise_wrapper,
                                  bitwise_check,
                                  &args_stu, &args_ref, BASELINE_BITWISE});
            break;
        }

        case KernelTarget::MatMul: {
            constexpr int n = 512;
            matmul_args args_stu, args_ref;
            initialize_matmul(args_stu, n, seed);
            initialize_matmul(args_ref, n, seed);
            std::println("\tMatMul: n={}", n);

            benchmarks.push_back({"MatMul (Student)",
                                  stu_matmul_wrapper,
                                  naive_matmul_wrapper,
                                  matmul_check,
                                  &args_stu, &args_ref, BASELINE_MATMUL});
            break;
        }

        case KernelTarget::TraceReplay: {
            trace_replay_args args_stu, args_ref;
            initialize_trace_replay(args_stu, 1 << 16, 1 << 20, seed);
            initialize_trace_replay(args_ref, 1 << 16, 1 << 20, seed);
            std::println("\tTrace Replay: records={}, trace_length={}",
                         args_stu.records.size(),
                         args_stu.trace.size());

            benchmarks.push_back({"Trace Replay (Student)",
                                  stu_trace_replay_wrapper,
                                  naive_trace_replay_wrapper,
                                  trace_replay_check,
                                  &args_stu, &args_ref, BASELINE_TRACE_REPLAY});
            break;
        }

        case KernelTarget::Graph: {
            constexpr std::size_t node_count = 1024000;
            constexpr int avg_degree = 8;
            graph_args args_stu, args_ref;
            initialize_graph(&args_stu, node_count, avg_degree, seed);
            initialize_graph(&args_ref, node_count, avg_degree, seed);
            std::println("\tGraph: node_count={}, avg_degree={}", node_count, avg_degree);

            benchmarks.push_back({"Graph (Student)",
                                  stu_graph_wrapper,
                                  naive_graph_wrapper,
                                  graph_check,
                                  &args_stu, &args_ref, BASELINE_GRAPH});
            break;
        }

        case KernelTarget::GRFF: {
            constexpr std::size_t grff_size = 1024000;
            grff_args args_stu, args_ref;
            initialize_grff(&args_stu, grff_size, seed);
            initialize_grff(&args_ref, grff_size, seed);
            std::println("\tGRFF: feature size={}", args_stu.a_features.size());

            benchmarks.push_back({"GRFF (Student)",
                                  stu_grff_wrapper,
                                  naive_grff_wrapper,
                                  grff_check,
                                  &args_stu, &args_ref, BASELINE_GRFF});
            break;
        }

        case KernelTarget::ImageProc: {
            constexpr std::size_t image_width = 1024;
            constexpr std::size_t image_height = 1000;
            image_proc_args args_stu, args_ref;
            initialize_image_proc(&args_stu, image_width, image_height, seed);
            initialize_image_proc(&args_ref, image_width, image_height, seed);
            std::println("\tImage Proc: {} x {}", args_stu.width, args_stu.height);

            benchmarks.push_back({"Image Proc (Student)",
                                  stu_image_proc_wrapper,
                                  naive_image_proc_wrapper,
                                  image_proc_check,
                                  &args_stu, &args_ref, BASELINE_IMAGE_PROC});
            break;
        }

        case KernelTarget::FilterGradient: {
            constexpr std::size_t WIDTH = 1024;
            constexpr std::size_t HEIGHT = 1024;
            filter_gradient_args args_stu, args_ref;
            initialize_filter_gradient(&args_stu, WIDTH, HEIGHT, seed);
            initialize_filter_gradient(&args_ref, WIDTH, HEIGHT, seed);
            std::println("\tFilter Gradient: {} x {}", HEIGHT, WIDTH);

            benchmarks.push_back({"Filter Gradient (Student)",
                                  stu_filter_gradient_wrapper,
                                  naive_filter_gradient_wrapper,
                                  filter_gradient_check,
                                  &args_stu, &args_ref, BASELINE_FILTER_GRADIENT});
            break;
        }
    }

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