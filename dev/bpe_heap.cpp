// bpe_heap.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <cstdint>
#include <string>
#include <tuple>

namespace py = pybind11;

// pack pair into uint64 key
static inline uint64_t pack_pair(uint32_t a, uint32_t b) {
    return ( (uint64_t)a << 32 ) | (uint64_t)b;
}
static inline std::pair<uint32_t,uint32_t> unpack_pair(uint64_t k) {
    return { (uint32_t)(k >> 32), (uint32_t)(k & 0xffffffffu) };
}

// heap node: count, tie, pair_key
struct HeapNode {
    int count;
    uint64_t tie;
    uint64_t pair_key;
};
struct HeapComp {
    bool operator()(HeapNode const& a, HeapNode const& b) const {
        if (a.count != b.count) return a.count < b.count; // max-heap by count
        return a.tie > b.tie;
    }
};

py::tuple train_bpe_heap(
    const std::vector<std::vector<int>>& seqs_in,
    const std::vector<std::string>& init_vocab_bytes,
    int num_merges
) {
    // Copy vocab bytes to mutable vector
    std::vector<std::string> vocab_bytes = init_vocab_bytes;
    // Node arrays
    std::vector<int> node_sym;    // token id (index into vocab_bytes)
    std::vector<int> node_prev;
    std::vector<int> node_next;
    std::vector<char> node_alive;
    std::vector<int> node2seq;
    std::vector<int> seq_heads;

    node_sym.reserve( (size_t)100000 );
    node_prev.reserve( (size_t)100000 );
    node_next.reserve( (size_t)100000 );
    node_alive.reserve( (size_t)100000 );
    node2seq.reserve( (size_t)100000 );

    // Build nodes from input sequences
    int node_id = 0;
    for (size_t sidx = 0; sidx < seqs_in.size(); ++sidx) {
        const auto &seq = seqs_in[sidx];
        if (seq.empty()) {
            seq_heads.push_back(-1);
            continue;
        }
        int head = node_id;
        seq_heads.push_back(head);
        for (size_t i = 0; i < seq.size(); ++i) {
            node_sym.push_back(seq[i]);
            node_prev.push_back( (int)(i==0 ? -1 : (node_id-1)) );
            node_next.push_back( (int)(i+1<seq.size() ? node_id+1 : -1) );
            node_alive.push_back(1);
            node2seq.push_back((int)sidx);
            ++node_id;
        }
    }

    // Build pair->occurrences (unordered_set of left node ids)
    std::unordered_map<uint64_t, std::unordered_set<int>> pair2occ;
    pair2occ.reserve(1<<16);
    for (int nid = 0; nid < node_id; ++nid) {
        int r = node_next[nid];
        if (r != -1) {
            uint32_t a = (uint32_t) node_sym[nid];
            uint32_t b = (uint32_t) node_sym[r];
            uint64_t key = pack_pair(a,b);
            pair2occ[key].insert(nid);
        }
    }

    // Build initial heap
    std::priority_queue<HeapNode, std::vector<HeapNode>, HeapComp> heap;
    uint64_t tie_counter = 0;
    for (auto &kv : pair2occ) {
        heap.push(HeapNode{ (int)kv.second.size(), tie_counter++, kv.first });
    }

    std::vector<std::pair<int,int>> merges_out;
    merges_out.reserve(std::max(0, num_merges));

    auto remove_occurrence = [&](uint64_t key, int left_nid) {
        auto it = pair2occ.find(key);
        if (it == pair2occ.end()) return;
        it->second.erase(left_nid);
        if (it->second.empty()) pair2occ.erase(it);
        else heap.push(HeapNode{ (int)it->second.size(), tie_counter++, key });
    };
    auto add_occurrence = [&](uint64_t key, int left_nid) {
        auto &s = pair2occ[key]; // creates if absent
        auto inserted = s.insert(left_nid);
        if (inserted.second) {
            heap.push(HeapNode{ (int)s.size(), tie_counter++, key });
        }
    };

    // merge main loop
    for (int merge_i = 0; merge_i < num_merges; ++merge_i) {
        // get best pair via lazy-pop
        HeapNode bestHN;
        bool found = false;
        while (!heap.empty()) {
            bestHN = heap.top(); heap.pop();
            auto it = pair2occ.find(bestHN.pair_key);
            int cur_count = (it==pair2occ.end() ? 0 : (int)it->second.size());
            if (cur_count == bestHN.count && cur_count > 0) {
                found = true;
                break;
            }
            // else stale, continue popping
        }
        if (!found) break; // no more pairs

        uint64_t pair_key = bestHN.pair_key;
        auto pa = unpack_pair(pair_key);
        uint32_t a = pa.first;
        uint32_t b = pa.second;

        // record merge pair indices for output (a,b are ints)
        merges_out.emplace_back((int)a, (int)b);

        // create new vocab token bytes by concatenation
        std::string new_bytes = vocab_bytes[a] + vocab_bytes[b];
        int new_id = (int)vocab_bytes.size();
        vocab_bytes.push_back(std::move(new_bytes));

        // take snapshot of occurrences vector to iterate
        auto occ_it = pair2occ.find(pair_key);
        if (occ_it == pair2occ.end()) continue;
        std::vector<int> occs;
        occs.reserve(occ_it->second.size());
        for (int left : occ_it->second) occs.push_back(left);
        // remove this pair from mapping now
        pair2occ.erase(occ_it);

        // process occurrences
        for (int left : occs) {
            if (left < 0 || left >= (int)node_sym.size()) continue;
            if (!node_alive[left]) continue;
            int right = node_next[left];
            if (right == -1 || !node_alive[right]) continue;
            if ((uint32_t)node_sym[left] != a || (uint32_t)node_sym[right] != b) continue; // changed by prior merges

            int left_prev = node_prev[left];
            int right_next = node_next[right];
            int seq_index = node2seq[left];

            // remove occurrences that involve left_prev, left, right
            if (left_prev != -1) {
                uint64_t k = pack_pair((uint32_t)node_sym[left_prev], (uint32_t)node_sym[left]);
                remove_occurrence(k, left_prev);
            }
            {
                uint64_t k = pack_pair((uint32_t)node_sym[left], (uint32_t)node_sym[right]);
                remove_occurrence(k, left);
            }
            if (right != -1 && right_next != -1) {
                uint64_t k = pack_pair((uint32_t)node_sym[right], (uint32_t)node_sym[right_next]);
                remove_occurrence(k, right);
            }

            // create new node
            int nid_new = (int)node_sym.size();
            node_sym.push_back(new_id);
            node_prev.push_back(left_prev);
            node_next.push_back(right_next);
            node_alive.push_back(1);
            node2seq.push_back(seq_index);

            // link neighbors
            if (left_prev != -1) node_next[left_prev] = nid_new;
            else seq_heads[seq_index] = nid_new;
            if (right_next != -1) node_prev[right_next] = nid_new;

            // mark old nodes dead
            node_alive[left] = 0;
            node_alive[right] = 0;

            // add occurrences for newly created neighbors
            if (left_prev != -1) {
                uint64_t k = pack_pair((uint32_t)node_sym[left_prev], (uint32_t)node_sym[nid_new]);
                add_occurrence(k, left_prev);
            }
            if (right_next != -1) {
                uint64_t k = pack_pair((uint32_t)node_sym[nid_new], (uint32_t)node_sym[right_next]);
                add_occurrence(k, nid_new);
            }
        } // for each occurrence
    } // merges

    // prepare return: vocab_bytes (as python bytes objects) and merges list
    py::list vocab_py(vocab_bytes.size());
    for (size_t i = 0; i < vocab_bytes.size(); ++i) {
        vocab_py[i] = py::bytes(vocab_bytes[i]);
    }

    py::list merges_py(merges_out.size());
    for (size_t i = 0; i < merges_out.size(); ++i) {
        merges_py[i] = py::make_tuple(merges_out[i].first, merges_out[i].second);
    }

    return py::make_tuple(vocab_py, merges_py);
}

PYBIND11_MODULE(bpe_cpp_heap, m) {
    m.doc() = "BPE incremental trainer: linked nodes + heap (C++/pybind11)";
    m.def("train_bpe_heap", &train_bpe_heap,
        "train_bpe_heap(seqs:list[list[int]], vocab_bytes:list[bytes], num_merges:int) -> (vocab_bytes_out, merges_pairs)");
}
