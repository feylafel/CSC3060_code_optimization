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

int main() {
    std::uint32_t seed = 12345u;
    std::vector<bench_t> benchmarks;

    // ============================================================
    // UNCOMMENT EXACTLY ONE KERNEL BLOCK BELOW (keep the others
    // fully commented, or you will get compile errors or wrong runs).
    // ============================================================

    /* ========== Black-Scholes ==========
    blackscholes_args bs_stu, bs_ref;
    initialize_blackscholes(bs_stu, 81920, seed);
    initialize_blackscholes(bs_ref, 81920, seed);
    std::println("\tBlack-Scholes options: {}", bs_stu.spot_price.size());
    benchmarks = {
        {"Black-Scholes (Student)", stu_BlkSchls_wrapper, naive_BlkSchls_wrapper,
         BlkSchls_check, &bs_stu, &bs_ref, BASELINE_BLACKSCHOLES},
    };
    ========== end ========== */

    /* ========== Sparse SpMM ==========
    sparse_spmm_args sp_stu, sp_ref;
    initialize_spmm(sp_stu, 512, 512, -1, {}, seed);
    initialize_spmm(sp_ref, 512, 512, -1, {}, seed);
    std::println("\tSparse A (CSR): {} x {}, nnz={}", sp_stu.csr.rows,
                 sp_stu.csr.cols, sp_stu.csr.values.size());
    benchmarks = {
        {"Sparse SpMM (Student)", stu_sparse_spmm_wrapper, naive_sparse_spmm_wrapper,
         sparse_spmm_check, &sp_stu, &sp_ref, BASELINE_SPARSE_SPMM},
    };
    ========== end ========== */

    /* ========== ReLU ==========
    constexpr std::size_t relu_n = 1024000;
    relu_args relu_stu, relu_ref;
    initialize_relu(&relu_stu, relu_n, seed);
    initialize_relu(&relu_ref, relu_n, seed);
    std::println("\tReLU: vector length={}", relu_n);
    benchmarks = {
        {"ReLU (Student)", stu_relu_wrapper, naive_relu_wrapper, relu_check,
         &relu_stu, &relu_ref, BASELINE_RELU},
    };
    ========== end ========== */

    /* ========== Bitwise ==========
    constexpr std::size_t bit_n = 1024000;
    bitwise_args bit_stu, bit_ref;
    initialize_bitwise(&bit_stu, bit_n, seed);
    initialize_bitwise(&bit_ref, bit_n, seed);
    std::println("\tBitwise: vector length={}", bit_n);
    benchmarks = {
        {"Bitwise (Student)", stu_bitwise_wrapper, naive_bitwise_wrapper,
         bitwise_check, &bit_stu, &bit_ref, BASELINE_BITWISE},
    };
    ========== end ========== */

    /* ========== MatMul ==========
    constexpr int mat_n = 512;
    matmul_args mm_stu, mm_ref;
    initialize_matmul(mm_stu, mat_n, seed);
    initialize_matmul(mm_ref, mat_n, seed);
    std::println("\tMatMul: n={}", mat_n);
    benchmarks = {
        {"MatMul (Student)", stu_matmul_wrapper, naive_matmul_wrapper, matmul_check,
         &mm_stu, &mm_ref, BASELINE_MATMUL},
    };
    ========== end ========== */

    /* ========== Trace replay ==========
    trace_replay_args tr_stu, tr_ref;
    initialize_trace_replay(tr_stu, 1 << 16, 1 << 20, seed);
    initialize_trace_replay(tr_ref, 1 << 16, 1 << 20, seed);
    std::println("\tTrace Replay: records={}, trace_length={}",
                 tr_stu.records.size(), tr_stu.trace.size());
    benchmarks = {
        {"Trace Replay (Student)", stu_trace_replay_wrapper, naive_trace_replay_wrapper,
         trace_replay_check, &tr_stu, &tr_ref, BASELINE_TRACE_REPLAY},
    };
    ========== end ========== */

    /* ========== Graph ==========
    constexpr std::size_t g_nodes = 1024000;
    constexpr int g_deg = 8;
    graph_args g_stu, g_ref;
    initialize_graph(&g_stu, g_nodes, g_deg, seed);
    initialize_graph(&g_ref, g_nodes, g_deg, seed);
    std::println("\tGraph: node_count={}, avg_degree={}", g_nodes, g_deg);
    benchmarks = {
        {"Graph (Student)", stu_graph_wrapper, naive_graph_wrapper, graph_check,
         &g_stu, &g_ref, BASELINE_GRAPH},
    };
    ========== end ========== */

    /* ========== GRFF ==========
    constexpr std::size_t grff_n = 1024000;
    grff_args grff_stu, grff_ref;
    initialize_grff(&grff_stu, grff_n, seed);
    initialize_grff(&grff_ref, grff_n, seed);
    std::println("\tGRFF: feature size={}", grff_stu.a_features.size());
    benchmarks = {
        {"GRFF (Student)", stu_grff_wrapper, naive_grff_wrapper, grff_check,
         &grff_stu, &grff_ref, BASELINE_GRFF},
    };
    ========== end ========== */

    /* ========== Image proc ==========
    constexpr std::size_t img_w = 1024, img_h = 1000;
    image_proc_args ip_stu, ip_ref;
    initialize_image_proc(&ip_stu, img_w, img_h, seed);
    initialize_image_proc(&ip_ref, img_w, img_h, seed);
    std::println("\tImage Proc: {} x {}", ip_stu.width, ip_stu.height);
    benchmarks = {
        {"Image Proc (Student)", stu_image_proc_wrapper, naive_image_proc_wrapper,
         image_proc_check, &ip_stu, &ip_ref, BASELINE_IMAGE_PROC},
    };
    ========== end ========== */

    // ========== Filter Gradient (EXAMPLE: ACTIVE) ==========
    constexpr std::size_t fg_w = 1024, fg_h = 1024;
    filter_gradient_args fg_stu, fg_ref;
    initialize_filter_gradient(&fg_stu, fg_w, fg_h, seed);
    initialize_filter_gradient(&fg_ref, fg_w, fg_h, seed);
    std::println("\tFilter Gradient: {} x {}", fg_h, fg_w);
    benchmarks = {
        {"Filter Gradient (Student)", stu_filter_gradient_wrapper,
         naive_filter_gradient_wrapper, filter_gradient_check, &fg_stu, &fg_ref,
         BASELINE_FILTER_GRADIENT},
    };
    // To use another kernel: comment this whole block with /* */ and
    // uncomment exactly one other block above.

    // ---------------- timing loop (do not change) ----------------
    std::cout << "\nRunning Benchmarks...\n";
    std::cout << "--------------------------------------------------------\n";
    std::cout << std::left << std::setw(25) << "Benchmark" << std::setw(12)
              << "Status" << std::right << std::setw(15) << "Nanoseconds" << "\n";
    std::cout << "--------------------------------------------------------\n";

    constexpr int k_best = 20;
    for (const auto& bench : benchmarks) {
        std::chrono::nanoseconds avg_time{0};
        for (int i = 0; i < k_best; ++i) {
            flush_cache();
            const auto elapsed = measure_time([&] { bench.tfunc(bench.args); });
            avg_time += elapsed;
        }
        avg_time /= static_cast<std::uint64_t>(k_best);

        const bool correct =
            bench.checkFunc(bench.args, bench.ref_args, bench.naiveFunc);

        std::cout << std::left << std::setw(25) << bench.description;
        if (!correct) {
            std::cout << "\033[1;31mFAILED\033[0m" << std::right
                      << std::setw(15) << "N/A" << "\n";
            std::cout << "  Error: Results do not match naive implementation!\n";
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