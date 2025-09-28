import os
import json
import time
import cProfile
import pstats
import sys
from memory_profiler import memory_usage
from pathlib import Path

# Assume the train_bpe function is in cs336_basics/bpe.py
# Please ensure your PYTHONPATH includes the directory containing cs336_basics
try:
    from bpe_vx_cpp import train_bpe
except ImportError:
    print("Error: Could not import train_bpe. Please ensure cs336_basics is in your PYTHONPATH.")
    print("You can try running: export PYTHONPATH=$PYTHONPATH:$(pwd)")
    sys.exit(1)

def main():
    """
    Trains a BPE tokenizer on the TinyStories dataset and saves the results.
    """
    # --- Configuration ---
    # !!!IMPORTANT!!! 
    # Please replace this path with the actual path to your TinyStories dataset file
    input_path = "/inspire/dataset/cs336/v1/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    output_dir = Path("bpe_tokenizer/tinystories")
    
    # Check if the input file exists
    if not Path(input_path).exists():
        print(f"Error: Input file '{input_path}' does not exist.")
        print("Please edit the script file 'train_bpe.py' and set the correct 'input_path'.")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab_file = output_dir / "vocab.json"
    merges_file = output_dir / "merges.txt"
    profile_stats_file = output_dir / "profile_stats.txt"

    # --- Training and Profiling ---
    print(f"Starting BPE training on '{input_path}'...")
    
    start_time = time.time()
    
    # Profile the training function
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Record memory usage
    mem_usage_before = memory_usage(max_usage=True)

    # Call the training function
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        num_processes=32
    )
    
    profiler.disable()
    
    end_time = time.time()
    mem_usage_after = memory_usage(max_usage=True)

    # --- Save Results ---
    print(f"Training complete. Saving results to '{output_dir}'...")

    # Save vocabulary
    # Decode byte strings to strings for JSON serialization
    serializable_vocab = {k: v.decode('utf-8', errors='ignore') for k, v in vocab.items()}
    with open(vocab_file, "w", encoding="utf-8") as f:
        json.dump(serializable_vocab, f, ensure_ascii=False, indent=2)

    # Save merge rules
    with open(merges_file, "w", encoding="utf-8") as f:
        for pair in merges:
            # Decode byte pair to strings
            token1 = pair[0].decode('utf-8', errors='ignore')
            token2 = pair[1].decode('utf-8', errors='ignore')
            f.write(f"{token1} {token2}\n")

    # Save profiling results
    with open(profile_stats_file, "w") as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.sort_stats("cumulative")
        stats.print_stats()

    print("Results saved.")

    # --- Answer Questions ---
    training_duration_seconds = end_time - start_time
    training_duration_minutes = training_duration_seconds / 60
    max_mem_usage_mb = max(mem_usage_before, mem_usage_after)

    # Find the longest token
    longest_token = b""
    if vocab:
        longest_token = max(vocab.values(), key=len)

    print("\n--- Experiment Results Analysis ---")
    print(f"1. Training time: {training_duration_seconds:.2f} seconds (~{training_duration_minutes:.2f} minutes).")
    print(f"   Peak memory usage: {max_mem_usage_mb:.2f} MB.")
    
    print(f"\n2. The longest token in the vocabulary is: {longest_token}")
    print(f"   Length: {len(longest_token)} bytes.")
    print("   This is often meaningful as it represents the most common and repetitive byte sequences in the dataset, such as common words or phrases.")

    print(f"\n3. Performance profiling results have been saved to '{profile_stats_file}'.")
    print("   By examining this file, you can identify the most time-consuming parts of the training process. Typically, pre-tokenization and initial pair counting are the main bottlenecks.")

if __name__ == "__main__":
    main()
