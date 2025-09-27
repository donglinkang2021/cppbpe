from . import bpe_cpp
sample = [[1,2,3,1,2],[1,2,1,2],[5,6,7]]
print(bpe_cpp.compute_pair_counts(sample))      # 期望 {(1,2):4, (2,3):1, ...}
print(bpe_cpp.apply_merge_all(sample, (1,2), 100))
# -> [[100,3,100],[100,100],[5,6,7]]
