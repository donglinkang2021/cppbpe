#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <unordered_map>
#include <vector>
#include <queue>
#include <cstdint>
#include <unordered_set>
#include <algorithm> // For std::max_element

namespace py = pybind11;

inline uint64_t make_key(int a, int b) {
    return (uint64_t(a) << 32) | uint32_t(b);
}

std::tuple<
    std::unordered_map<int, std::vector<uint8_t>>,
    std::vector<std::pair<int,int>>
> train_bpe_core(
    std::vector<std::vector<int>> ids,
    int vocab_size,
    std::unordered_map<int, std::vector<uint8_t>> vocab
) {
    std::unordered_map<uint64_t, int> counts;
    std::unordered_map<uint64_t, std::unordered_set<int>> pair_to_indices;

    // 1. Initial pair statistics
    for (size_t i = 0; i < ids.size(); i++) {
        auto& token_ids = ids[i];
        for (size_t j = 0; j + 1 < token_ids.size(); j++) {
            uint64_t key = make_key(token_ids[j], token_ids[j+1]);
            counts[key]++;
            pair_to_indices[key].insert(i);
        }
    }

    std::vector<std::pair<int,int>> merges;
    int next_id = 256;
    while (vocab.count(next_id)) {
        next_id++;
    }

    int num_merges = vocab_size - vocab.size();
    for (int m = 0; m < num_merges; m++) {
        if (counts.empty()) break;

        // Find the most frequent pair by iterating through the counts map, similar to Python's max(counts, key=rank)
        auto max_it = std::max_element(counts.begin(), counts.end(),
            [&](const auto& a, const auto& b) {
                if (a.second != b.second) {
                    return a.second < b.second; // Compare frequency
                }
                // Tie-breaking: compare pairs by their byte values
                int a1 = int(a.first >> 32);
                int b1 = int(a.first & 0xffffffff);
                int a2 = int(b.first >> 32);
                int b2 = int(b.first & 0xffffffff);
                const auto& v_a1 = vocab.at(a1);
                const auto& v_a2 = vocab.at(a2);
                if (v_a1 != v_a2) {
                    return v_a1 < v_a2;
                }
                const auto& v_b1 = vocab.at(b1);
                const auto& v_b2 = vocab.at(b2);
                return v_b1 < v_b2;
            });

        uint64_t key = max_it->first;
        int a = int(key >> 32);
        int b = int(key & 0xffffffff);

        // Create new token
        std::vector<uint8_t> new_tok = vocab[a];
        new_tok.insert(new_tok.end(), vocab[b].begin(), vocab[b].end());
        vocab[next_id] = new_tok;
        merges.push_back({a, b});

        // Update affected sequences (incremental update)
        auto affected_indices = pair_to_indices[key];
        
        for (int idx : affected_indices) {
            auto& seq = ids[idx];
            if (seq.size() < 2) continue;
            
            // Decrement counts for pairs in the old sequence
            for (size_t i = 0; i + 1 < seq.size(); ++i) {
                uint64_t old_key = make_key(seq[i], seq[i+1]);
                if (counts.count(old_key)) {
                    counts[old_key]--;
                    pair_to_indices[old_key].erase(idx);
                    if (counts[old_key] == 0) {
                        counts.erase(old_key);
                        pair_to_indices.erase(old_key);
                    }
                }
            }

            // Create new sequence by merging the pair
            std::vector<int> new_seq;
            new_seq.reserve(seq.size());
            for (size_t i = 0; i < seq.size();) {
                if (i + 1 < seq.size() && seq[i] == a && seq[i+1] == b) {
                    new_seq.push_back(next_id);
                    i += 2;
                } else {
                    new_seq.push_back(seq[i]);
                    i++;
                }
            }
            seq = std::move(new_seq);

            // Increment counts for pairs in the new sequence
            if (seq.size() < 2) continue;
            for (size_t i = 0; i + 1 < seq.size(); ++i) {
                uint64_t new_key = make_key(seq[i], seq[i+1]);
                counts[new_key]++;
                pair_to_indices[new_key].insert(idx);
            }
        }
        
        // Clean up the merged pair
        counts.erase(key);
        pair_to_indices.erase(key);

        next_id++;
        while (vocab.count(next_id)) {
            next_id++;
        }
    }

    return {vocab, merges};
}

PYBIND11_MODULE(bpe_core, m) {
    m.def("train_bpe_core", &train_bpe_core,
          "Train BPE merges in C++");
}
