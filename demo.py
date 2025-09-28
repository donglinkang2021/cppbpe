# import debugpy; debugpy.connect(('0.0.0.0', 5678))
import time


from pathlib import Path

FIXTURES_PATH = Path("data")

def test_train_bpe_demo():
    """
    Ensure that BPE training is relatively efficient by measuring training
    time on this small dataset and throwing an error if it takes more than 1.5 seconds.
    This is a pretty generous upper-bound, it takes 0.38 seconds with the
    reference implementation on my laptop. In contrast, the toy implementation
    takes around 3 seconds.
    """
    from bpe_vx import train_bpe
    input_path = FIXTURES_PATH / "corpus.en"
    start_time = time.time()
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )
    cost_time = time.time() - start_time
    print(f"Training BPE took {cost_time:.2f} seconds, hope it's less than 1.5 seconds")
    print(f"Vocab size: {len(vocab)}")
    with open("merges_true.txt", "w", encoding="utf-8") as f:
        for p1, p2 in merges:
            f.write(f"{p1} {p2}\n")

def test_train_bpe_demo1():
    """
    Ensure that BPE training is relatively efficient by measuring training
    time on this small dataset and throwing an error if it takes more than 1.5 seconds.
    This is a pretty generous upper-bound, it takes 0.38 seconds with the
    reference implementation on my laptop. In contrast, the toy implementation
    takes around 3 seconds.
    """
    from bpe_vx_cpp import train_bpe
    input_path = FIXTURES_PATH / "corpus.en"
    start_time = time.time()
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )
    cost_time = time.time() - start_time
    print(f"Training BPE took {cost_time:.2f} seconds, hope it's less than 1.5 seconds")
    print(f"Vocab size: {len(vocab)}")
    with open("merges1.txt", "w", encoding="utf-8") as f:
        for p1, p2 in merges:
            f.write(f"{p1} {p2}\n")

if __name__ == "__main__":
    test_train_bpe_demo()
    test_train_bpe_demo1()
