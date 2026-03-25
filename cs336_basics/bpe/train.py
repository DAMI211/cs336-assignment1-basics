import os
from datetime import datetime
from multiprocessing import Pool
from cs336_basics.bpe.pretokenization import _find_chunk_boundaries, _count_file_chunk


def _log(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


def _validate_vocab_size(vocab_size: int, num_special_tokens: int) -> None:
    initial_size = 256 + num_special_tokens
    if initial_size > vocab_size:
        raise ValueError(
            f"vocab_size ({vocab_size}) is too small: "
            f"256 base bytes + {num_special_tokens} special tokens = {initial_size}."
        )


def _init_base_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    vocab = {i: token.encode("utf-8") for i, token in enumerate(special_tokens)}
    for i in range(256):
        vocab[len(special_tokens) + i] = bytes([i])
    return vocab


class _RevBytes:
    __slots__ = ("b",)
    def __init__(self, b: bytes): self.b = b
    def __lt__(self, other): return self.b > other.b # 不管 bytes 长度如何，整个比较逻辑完全反转，没有副作用。
    def __eq__(self, other): return self.b == other.b


def _heap_entry(freq: int, pair: tuple[bytes, bytes]) -> tuple:
    return (-freq, _RevBytes(pair[0]), _RevBytes(pair[1]), pair)


def _compute_bpe_merges(                                                                                                                                                                                                                          
    pretoken_counts: dict[str, int],                                                                                                                                                                                                              
    vocab: dict[int, bytes],                                                                                                                                                                                                                      
    merge_times: int,                                                                                                                                                                                                                             
) -> list[tuple[bytes, bytes]]:         
    import heapq
    from collections import defaultdict

    merges: list[tuple[bytes, bytes]] = []

    # 1. 初始化数据结构
    unique_pretokens = list(pretoken_counts.keys())
    counts = [pretoken_counts[p] for p in unique_pretokens]
    
    # word_sequences[word_idx] = list of token_bytes
    word_sequences = []
    for p in unique_pretokens:
        word_sequences.append([bytes([b]) for b in p.encode("utf-8")])
        
    pair_freq = defaultdict(int)
    # pair_to_word_indices: dict[pair, set[word_idx]]
    # 记录每个 pair 出现在哪些唯一的预分词索引中
    pair_to_word_indices = defaultdict(set)
    
    for i, seq in enumerate(word_sequences):
        count = counts[i]
        for j in range(len(seq) - 1):
            pair = (seq[j], seq[j+1])
            pair_freq[pair] += count
            pair_to_word_indices[pair].add(i)
            
    # 最大堆（负频次模拟），元素: (-freq, _RevBytes(p1), _RevBytes(p2), pair)
    heap = [_heap_entry(freq, pair) for pair, freq in pair_freq.items() if freq > 0]
    heapq.heapify(heap)
    
    # pair_freq  →  权威数据源（始终是最新频次）
    # heap       →  加速查找用的索引（可能有过期条目）
    for i in range(merge_times):
        # 弹出最优 pair，跳过过期条目（lazy deletion）
        best_pair = None
        while heap:
            neg_freq, _, _, candidate = heapq.heappop(heap)
            if pair_freq.get(candidate, 0) == -neg_freq and -neg_freq > 0:
                best_pair = candidate
                break

        if best_pair is None:
            break

        if (i + 1) % 100 == 0:
            _log(f"Merge {i + 1}/{merge_times}: best_pair={best_pair}, freq={-neg_freq}")

        merges.append(best_pair)
        new_token = best_pair[0] + best_pair[1]
        vocab[max(vocab) + 1] = new_token

        # 仅处理包含 best_pair 的预分词序列
        affected_indices = pair_to_word_indices[best_pair]
        changed_pairs = set()
        
        # 必须遍历索引集的副本，因为在循环中会修改 pair_to_word_indices
        for word_idx in list(affected_indices):
            seq = word_sequences[word_idx]
            count = counts[word_idx]
            
            # 1. 从全局频率和索引中移除该单词的所有旧 pair
            for j in range(len(seq) - 1):
                p = (seq[j], seq[j+1])
                pair_freq[p] -= count
                pair_to_word_indices[p].discard(word_idx)
                changed_pairs.add(p)
            
            # 2. 执行合并，生成新序列
            new_seq = []
            j = 0
            while j < len(seq):
                if j < len(seq) - 1 and (seq[j], seq[j+1]) == best_pair:
                    new_seq.append(new_token)
                    j += 2
                else:
                    new_seq.append(seq[j])
                    j += 1
            word_sequences[word_idx] = new_seq
            
            # 3. 将新序列的所有新 pair 添加到全局频率和索引中
            for j in range(len(new_seq) - 1):
                p = (new_seq[j], new_seq[j+1])
                pair_freq[p] += count
                pair_to_word_indices[p].add(word_idx)
                changed_pairs.add(p)

        # 更新堆：任何频率发生变化的 pair 都需要重新入堆
        # lazy deletion 会处理掉旧的错误频次条目
        for p in changed_pairs:
            freq = pair_freq.get(p, 0)
            if freq > 0:
                heapq.heappush(heap, _heap_entry(freq, p))
            elif p in pair_freq:
                del pair_freq[p]
            
            if p in pair_to_word_indices and not pair_to_word_indices[p]:
                del pair_to_word_indices[p]

    return merges


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    _validate_vocab_size(vocab_size, len(special_tokens))
    _log("Vocab size validated.")

    vocab = _init_base_vocab(special_tokens)
    _log(f"Base vocab initialized with {len(vocab)} tokens.")

    num_processes = os.cpu_count() or 1
    split_token = special_tokens[0].encode("utf-8") if special_tokens else b"\n"
    with open(input_path, "rb") as f:
        boundaries = _find_chunk_boundaries(f, num_processes, split_token)
    _log(f"Chunk boundaries found. Split file into {len(boundaries)-1} chunks.")

    args = [
        (input_path, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    with Pool(processes=num_processes) as pool:
        partial_counts = pool.map(_count_file_chunk, args)
    _log("Parallel counting of tokens completed.")

    pretoken_counts: dict[str, int] = {}
    for counts in partial_counts:
        for token, count in counts.items():
            pretoken_counts[token] = pretoken_counts.get(token, 0) + count
    _log(f"Aggregated pretoken counts. Found {len(pretoken_counts)} unique pretokens.")

    num_merges = vocab_size - len(special_tokens) - 256
    _log(f"Starting BPE merges (total: {num_merges})...")
    merges = _compute_bpe_merges(pretoken_counts, vocab, num_merges)
    _log("BPE merges completed.")

    return vocab, merges


if __name__ == "__main__":
    input_path = "/Users/dami/cs336/cs336-assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt"
    vocab_size = 100000
    special_tokens = ["<|endoftext|>"]
    train_bpe(input_path, vocab_size, special_tokens)
