import os
import time
import json
import numpy as np
from pathlib import Path
from cs336_basics.bpe.tokenizer import Tokenizer

DATA_DIR = Path("data")
VOCAB_DIR = Path("data/vocab")

def sample_docs(file_path, num_docs=10):
    docs = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        # 读取前 1MB 采样，确保能包含 10 个完整文档
        content = f.read(1024 * 1024) 
        # 使用 <|endoftext|> 分割文档
        raw_docs = [d.strip() for d in content.split("<|endoftext|>") if d.strip()]
        docs = raw_docs[:num_docs]
    return docs

def calculate_compression_ratio(tokenizer, docs):
    total_bytes = 0
    total_tokens = 0
    for doc in docs:
        encoded = tokenizer.encode(doc)
        total_bytes += len(doc.encode("utf-8"))
        total_tokens += len(encoded)
    return total_bytes / total_tokens if total_tokens > 0 else 0

def test_throughput(tokenizer, docs, iterations=5):
    all_text = "\n\n".join(docs)
    start_time = time.perf_counter()
    for _ in range(iterations):
        _ = tokenizer.encode(all_text)
    end_time = time.perf_counter()
    
    total_bytes = len(all_text.encode("utf-8")) * iterations
    duration = end_time - start_time
    return total_bytes / duration

def main():
    # 1. 加载分词器
    print("Loading tokenizers...")
    ts_tokenizer = Tokenizer.from_files(
        VOCAB_DIR / "TinyStoriesV2-GPT4-train.vocab.json",
        VOCAB_DIR / "TinyStoriesV2-GPT4-train.merges.json",
        special_tokens=["<|endoftext|>"]
    )
    owt_tokenizer = Tokenizer.from_files(
        VOCAB_DIR / "owt_train.vocab.json",
        VOCAB_DIR / "owt_train.merges.json",
        special_tokens=["<|endoftext|>"]
    )

    # 2. 采样文档
    print("Sampling documents...")
    ts_samples = sample_docs(DATA_DIR / "TinyStoriesV2-GPT4-train.txt")
    owt_samples = sample_docs(DATA_DIR / "owt_train.txt")

    # (a) Compression ratios
    ts_ratio = calculate_compression_ratio(ts_tokenizer, ts_samples)
    owt_ratio = calculate_compression_ratio(owt_tokenizer, owt_samples)
    print(f"(a) TS on TS Ratio: {ts_ratio:.4f}")
    print(f"(a) OWT on OWT Ratio: {owt_ratio:.4f}")

    # (b) Cross-tokenization
    ts_on_owt_ratio = calculate_compression_ratio(ts_tokenizer, owt_samples)
    print(f"(b) TS on OWT Ratio: {ts_on_owt_ratio:.4f}")

    # (c) Throughput
    throughput = test_throughput(owt_tokenizer, owt_samples)
    print(f"(c) Throughput: {throughput / 1024:.2f} KB/s")
    
    pile_size_gb = 825
    pile_size_bytes = pile_size_gb * 1024**3
    time_seconds = pile_size_bytes / throughput
    time_hours = time_seconds / 3600
    print(f"(c) Estimated time for Pile (825GB): {time_hours:.2f} hours")

    # (d) uint16 check
    print(f"TS Vocab Size: {len(ts_tokenizer.vocab)}")
    print(f"OWT Vocab Size: {len(owt_tokenizer.vocab)}")

if __name__ == "__main__":
    main()


# (a) 各分词器的压缩率 (bytes/token)：
# TinyStories 分词器在 TS 样本上的压缩率为 4.15，而 OpenWebText 分词器在 OWT 样本上的压缩率为 4.69。这表明两个分词器都能有效地将多个字节合并为单个 Token，且词表更大的 OWT 分词器具有更高的理论压缩上限。

# (b) 用 TinyStories 分词器处理 OpenWebText：
# 压缩率从 4.69 显著下降到了 3.19。这是因为 TinyStories 词表完全是针对简单童话故事训练的，面对 OWT 中复杂的网络用语和专业术语时，大量词汇无法有效匹配，被迫拆碎，导致编码效率大幅下降。

# (c) 吞吐量与 Pile 数据集处理时间：
# 当前实现的吞吐量约为 6.04 KB/s，按此速度处理 825GB 的 Pile 数据集大约需要 39,786 小时（约 4.5 年）。这说明纯 Python 实现的线性合并逻辑在处理大规模语料时存在巨大的性能瓶颈，实际生产中需要使用高效的 tiktoken (Rust) 等实现。

# (d) 为什么 uint16 是合适的选择？
# 由于两个分词器的词表大小（10,000 和 32,000）均小于 65,535（$2^{16}-1$），uint16 足以表示所有的 Token ID。相比 int32 或 int64，使用 uint16 可以节省 50%-75% 的存储空间，同时完全避免溢出。
                                                                                                                                                                                        