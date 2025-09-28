import os
import regex as re
from typing import BinaryIO
from multiprocessing import Pool, get_context
from collections import defaultdict

GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_COMPILED_PAT = re.compile(GPT2_PAT)

# uv run pytest tests/test_train_bpe.py
import bpe_core

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = 8
):
    vocab = {i: [i] for i in range(256)}
    for tok in special_tokens:
        vocab[len(vocab)] = list(tok.encode("utf-8"))

    with open(input_path, "rb") as f:
        bounds = find_chunk_boundaries(f, num_processes, "<|endoftext|>".encode("utf-8"))
    task_args = [(input_path, start, end, special_tokens)
                 for start, end in zip(bounds[:-1], bounds[1:])]
    with get_context("forkserver").Pool(processes=num_processes) as pool:
        chunk_results = pool.map(process_chunk, task_args)

    ids = [tok_ids for chunk in chunk_results for tok_ids in chunk]
    vocab, merges = bpe_core.train_bpe_core(ids, vocab_size, vocab)
    vocab = {i: bytes(v) for i, v in vocab.items()}
    return vocab, [(vocab[a], vocab[b]) for a, b in merges]

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    assert isinstance(split_special_token, bytes), \
        "Must represent special token as a bytestring"
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    chunk_size = max(1, file_size // desired_num_chunks)
    bounds = [i * chunk_size for i in range(desired_num_chunks + 1)]
    bounds[-1] = file_size
    mini = 4096  # 4k scan step (bigger save syscall)
    for bi in range(1, len(bounds) - 1):
        pos = bounds[bi]
        file.seek(pos)
        while True:
            buf = file.read(mini)
            if not buf:
                bounds[bi] = file_size
                break
            found = buf.find(split_special_token)
            if found != -1:
                bounds[bi] = pos + found
                break
            pos += len(buf)
    return sorted(set(bounds))


def process_chunk(args: tuple[str, int, int, list[str]]) -> list[list[int]]:
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as file:
        file.seek(start)
        chunk = file.read(end - start).decode("utf-8", errors="ignore")
    # 1. Remove special tokens by splitting the chunk at those tokens
    pattern = "|".join(re.escape(tok) for tok in special_tokens)
    documents = re.split(pattern, chunk)
    # 2. Pre-tokenize and count byte pair frequencies
    chunk_ids: list[list[int]] = []
    for doc in documents:
        tokens = [match.group(0).encode("utf-8") for match in _COMPILED_PAT.finditer(doc)]
        chunk_ids.extend([list(token) for token in tokens]) # list(bytes) -> list[int]
    return chunk_ids
