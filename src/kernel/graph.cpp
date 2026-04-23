#include "graph.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

void convert_graph(Graph &graph, StuGraph &stugraph)
{
    stugraph.n = graph.n;
    stugraph.nodepos.resize(graph.n + 1);
    for (int u = 0; u < graph.n; ++u) {
        stugraph.nodepos[u] = stugraph.dest.size();
        const Edge* e = graph.nodes[u].edges;
        while (e) {
            stugraph.dest.push_back(e->to);
            e = e->next;
        }
    }
    stugraph.nodepos[graph.n] = stugraph.dest.size();
}

void initialize_graph(graph_args* args,
                       std::size_t node_count,
                       int avg_degree,
                       std::uint_fast64_t seed) {
    if (!args) {
        return;
    }

    std::mt19937_64 gen(seed);
    std::uniform_int_distribution<int> dist(0, static_cast<int>(node_count) - 1);

    args->nodes.assign(node_count, Node{nullptr});
    args->edge_storage.clear();
    args->edge_storage.resize(node_count * static_cast<std::size_t>(avg_degree));

    args->graph.n = static_cast<int>(node_count);
    args->graph.nodes = args->nodes.data();

    std::size_t edge_pos = 0;

    for (std::size_t u = 0; u < node_count; ++u) {
        std::vector<int> neighbors;
        neighbors.reserve(avg_degree);

        for (int k = 0; k < avg_degree; ++k) {
            neighbors.push_back(dist(gen));
        }

        Edge* head = nullptr;
        for (int k = avg_degree - 1; k >= 0; --k) {
            Edge& e = args->edge_storage[edge_pos + static_cast<std::size_t>(k)];
            e.to = neighbors[static_cast<std::size_t>(k)];
            e.next = head;
            head = &e;
        }

        args->nodes[u].edges = head;
        edge_pos += static_cast<std::size_t>(avg_degree);
    }

    convert_graph(args->graph, args->stugraph);
    args->out = 0;
}

void naive_graph(std::uint64_t& out, const Graph& graph) {
    std::uint64_t checksum = 0;
    for (int u = 0; u < graph.n; ++u) {
        const Edge* e = graph.nodes[u].edges;
        while (e) {
            checksum += static_cast<std::uint64_t>(e->to);
            e = e->next;
        }
    }
    out = checksum;
}

void stu_graph(std::uint64_t& out, const StuGraph& stu_graph) {
    // TODO: You may need to add a function to convert data structure (not
    // included in time measurement), then implement your version in
    // stu_graph, whch is called by stu_graph_wrapper.  
    std::uint64_t res0 = 0;
    std::uint64_t res1 = 0;
    std::uint64_t res2 = 0;
    std::uint64_t res3 = 0;
    const uint32_t* __restrict nodepos = &stu_graph.nodepos[0];
    const uint32_t* __restrict dest = &stu_graph.dest[0];
    const int n = stu_graph.n;
    for (int u = 0; u < n; ++u){
        const uint32_t* r = dest + nodepos[u + 1];
        const uint32_t* p = dest + nodepos[u];
        for (; p + 7 < r; p += 8) {
            // simulate eight edges u -> v, where v == p[0], p[1], until p[7]
            res0 += p[0];
            res1 += p[1];
            res2 += p[2];
            res3 += p[3];
            res0 += p[4];
            res1 += p[5];
            res2 += p[6];
            res3 += p[7];
        }
        for (; p < r; ++p) {
            res0 += *p;
        }
    }
    out = res0 + res1 + res2 + res3;
}

void naive_graph_wrapper(void* ctx) {
    auto& args = *static_cast<graph_args*>(ctx);
    naive_graph(args.out, args.graph);
}

void stu_graph_wrapper(void* ctx) {
    auto& args = *static_cast<graph_args*>(ctx);
    stu_graph(args.out, args.stugraph);
}

bool graph_check(void* stu_ctx, void* ref_ctx, lab_test_func naive_func) {
    naive_func(ref_ctx);

    auto& stu_args = *static_cast<graph_args*>(stu_ctx);
    auto& ref_args = *static_cast<graph_args*>(ref_ctx);
    const auto eps = ref_args.epsilon;

    const double s = static_cast<double>(stu_args.out);
    const double r = static_cast<double>(ref_args.out);
    const double err = std::abs(s - r);
    const double atol = 0.0;
    const double rel = (std::abs(r) > 1e-12) ? err / std::abs(r) : err;

    debug_log("\tDEBUG: graph stu={} ref={} err={} rel={}\n",
              stu_args.out,
              ref_args.out,
              err,
              rel);

    return err <= (atol + eps * std::abs(r));
}
