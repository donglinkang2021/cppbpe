import numpy as np
from pathlib import Path
from tokenizers import Tokenizer
from tqdm import tqdm

def encode_to_bin_streaming(
    tokenizer: Tokenizer,
    input_path: Path,
    output_path: Path,
    chunk_lines: int = 50000,
    dtype=np.uint16,
    resume: bool = True,
):
    """
    分块读取 input_path，批量编码后写入 output_path（二进制连续 uint16）。
    如果 resume=True 且 output_path 存在，则跳过已写部分（基于文件大小推断）。
    """
    # 如果要 resumable，则检查已有 output_path 的字节大小
    start_token_count = 0
    if resume and output_path.exists():
        existing_size = output_path.stat().st_size
        # 每个 token 占 dtype 的字节数
        byte_per = np.dtype(dtype).itemsize
        if existing_size % byte_per != 0:
            print(f"Warning: existing file size {existing_size} is not multiple of {byte_per}")
        start_token_count = existing_size // byte_per
        print(f"Resuming: existing {start_token_count} tokens already written.")

    # 打开输出文件（append 模式）
    fout = output_path.open("ab")  # append in binary
    
    total_written = start_token_count

    # First, count total lines for tqdm
    print(f"Counting lines in {input_path}...")
    with input_path.open("r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    with input_path.open("r", encoding="utf-8") as fin:
        lines = []
        # Wrap file iterator with tqdm for progress visualization
        for line in tqdm(fin, total=total_lines, desc=f"Encoding {input_path.name}", unit=" lines"):
            lines.append(line)
            if len(lines) >= chunk_lines:
                # 批量编码
                encodings = tokenizer.encode_batch(lines)
                for enc in encodings:
                    ids = enc.ids
                    if ids:
                        arr = np.array(ids, dtype=dtype)
                        arr.tofile(fout)
                        total_written += arr.size
                lines = []

        # 最后一块残余
        if lines:
            encodings = tokenizer.encode_batch(lines)
            for enc in encodings:
                ids = enc.ids
                if ids:
                    arr = np.array(ids, dtype=dtype)
                    arr.tofile(fout)
                    total_written += arr.size

    fout.close()
    print(f"Finished encoding. Total tokens written: {total_written}")
    return total_written


def encode_to_bin_memmap(
    tokenizer: Tokenizer,
    input_path: Path,
    output_path: Path,
    chunk_lines: int = 50000,
    dtype = np.uint16,
):
    """
    使用两遍扫描 + memmap 预分配方式先统计 token 数量，再分块写入 memmap。
    适用于你能容忍两次扫描 input 的场景。
    """
    # Count total lines for tqdm progress bars
    print(f"Counting lines in {input_path} for progress estimation...")
    with input_path.open("r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    # --- 第一遍扫描，统计 token 总数 ---
    print("First pass: counting total tokens ...")
    total_tokens = 0
    with input_path.open("r", encoding="utf-8") as fin:
        lines = []
        for line in tqdm(fin, total=total_lines, desc=f"Pass 1/2 (count) {input_path.name}", unit=" lines"):
            lines.append(line)
            if len(lines) >= chunk_lines:
                encs = tokenizer.encode_batch(lines)
                for enc in encs:
                    total_tokens += len(enc.ids)
                lines = []
        if lines:
            encs = tokenizer.encode_batch(lines)
            for enc in encs:
                total_tokens += len(enc.ids)

    print(f"Estimated total tokens: {total_tokens}")

    # --- 用 memmap 预分配输出文件 ---
    print("Allocating memmap and writing tokens ...")
    mm = np.memmap(output_path, dtype=dtype, mode="w+", shape=(total_tokens,))
    write_pos = 0

    with input_path.open("r", encoding="utf-8") as fin:
        lines = []
        for line in tqdm(fin, total=total_lines, desc=f"Pass 2/2 (write) {input_path.name}", unit=" lines"):
            lines.append(line)
            if len(lines) >= chunk_lines:
                encs = tokenizer.encode_batch(lines)
                for enc in encs:
                    ids = enc.ids
                    if ids:
                        arr = np.array(ids, dtype=dtype)
                        n = arr.size
                        mm[write_pos : write_pos + n] = arr
                        write_pos += n
                lines = []
        if lines:
            encs = tokenizer.encode_batch(lines)
            for enc in encs:
                ids = enc.ids
                if ids:
                    arr = np.array(ids, dtype=dtype)
                    n = arr.size
                    mm[write_pos : write_pos + n] = arr
                    write_pos += n

    mm.flush()
    print(f"Finished memmap write. Total positions written: {write_pos}")
    return write_pos


def process_train_val(
    tokenizer_path: Path,
    train_input: Path,
    val_input: Path,
    output_dir: Path,
    chunk_lines: int = 50000,
    use_memmap: bool = False,
):
    # load tokenizer
    print(f"Loading tokenizer from {tokenizer_path} ...")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    print("Tokenizer loaded, vocab size:", tokenizer.get_vocab_size())

    output_dir.mkdir(parents=True, exist_ok=True)

    train_out = output_dir / "train.bin"
    val_out = output_dir / "val.bin"

    if use_memmap:
        print("Encoding train (memmap mode)...")
        cnt_train = encode_to_bin_memmap(tokenizer, train_input, train_out, chunk_lines=chunk_lines)
        print("Encoding val (memmap mode)...")
        cnt_val = encode_to_bin_memmap(tokenizer, val_input, val_out, chunk_lines=chunk_lines)
    else:
        print("Encoding train (streaming mode)...")
        cnt_train = encode_to_bin_streaming(tokenizer, train_input, train_out, chunk_lines=chunk_lines)
        print("Encoding val (streaming mode)...")
        cnt_val = encode_to_bin_streaming(tokenizer, val_input, val_out, chunk_lines=chunk_lines)

    print("Done. Train tokens:", cnt_train, "Val tokens:", cnt_val)


def main():
    # tokenizer_path = Path("bpe_tokenizer_hf/openwebtext/tokenizer.json")
    # train_input_path = Path("/inspire/dataset/cs336/v1/owt_train.txt")
    # val_input_path = Path("/inspire/dataset/cs336/v1/owt_valid.txt")
    # output_dir = Path("/inspire/hdd/global_user/donglinkang-253108120084/standford-cs336/assignment1-basics/data/openwebtext")
    
    tokenizer_path = Path("bpe_tokenizer_hf/tinystories/tokenizer.json")
    train_input_path = Path("/inspire/dataset/cs336/v1/TinyStoriesV2-GPT4-train.txt")
    val_input_path = Path("/inspire/dataset/cs336/v1/TinyStoriesV2-GPT4-valid.txt")
    output_dir = Path("/inspire/hdd/global_user/donglinkang-253108120084/standford-cs336/assignment1-basics/data/tinystories")

    # 可调参数
    chunk_lines = 50000        # 每块处理的行数，根据你机器内存 / tokenizer 性能调节
    use_memmap = False         # 是否用 memmap 模式（预分配 + 两遍扫描）

    process_train_val(
        tokenizer_path,
        train_input_path,
        val_input_path,
        output_dir,
        chunk_lines=chunk_lines,
        use_memmap=use_memmap,
    )


if __name__ == "__main__":
    main()
