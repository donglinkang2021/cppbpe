// bpe_cpp.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace py = pybind11;

// Pack two 32-bit ints into one 64-bit key
static inline uint64_t pack_pair(uint32_t a, uint32_t b) {
    return ( (uint64_t)a << 32 ) | (uint64_t)b;
}

py::dict compute_pair_counts(const std::vector<std::vector<int>>& seqs) {
    std::unordered_map<uint64_t, int> counts;
    counts.reserve(1 << 16);

    for (const auto& seq : seqs) {
        size_t L = seq.size();
        if (L < 2) continue;
        for (size_t i = 0; i + 1 < L; ++i) {
            uint32_t a = static_cast<uint32_t>(seq[i]);
            uint32_t b = static_cast<uint32_t>(seq[i+1]);
            uint64_t key = pack_pair(a, b);
            ++counts[key];
        }
    }

    py::dict out;
    for (auto &kv : counts) {
        uint64_t key = kv.first;
        int cnt = kv.second;
        uint32_t a = static_cast<uint32_t>(key >> 32);
        uint32_t b = static_cast<uint32_t>(key & 0xffffffffULL);
        out[ py::make_tuple((int)a, (int)b) ] = cnt;
    }
    return out;
}

std::vector<std::vector<int>> apply_merge_all(
    const std::vector<std::vector<int>>& seqs,
    std::pair<int,int> pair,
    int new_id
) {
    uint32_t a = static_cast<uint32_t>(pair.first);
    uint32_t b = static_cast<uint32_t>(pair.second);

    std::vector<std::vector<int>> out;
    out.reserve(seqs.size());

    for (const auto& seq : seqs) {
        std::vector<int> out_seq;
        out_seq.reserve(seq.size());
        size_t i = 0;
        const size_t L = seq.size();
        while (i < L) {
            if (i + 1 < L && static_cast<uint32_t>(seq[i]) == a && static_cast<uint32_t>(seq[i+1]) == b) {
                out_seq.push_back(new_id);
                i += 2;
            } else {
                out_seq.push_back(seq[i]);
                ++i;
            }
        }
        out.emplace_back(std::move(out_seq));
    }
    return out;
}

PYBIND11_MODULE(bpe_cpp, m) {
    m.doc() = "High-performance helpers for BPE: pair counting and bulk merge.";
    m.def("compute_pair_counts", &compute_pair_counts, "Compute pair counts from sequences");
    m.def("apply_merge_all", &apply_merge_all, "Apply a pair merge across all sequences");
}
