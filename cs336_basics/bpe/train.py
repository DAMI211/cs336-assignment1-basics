import os
import regex as re

# GPT-2 style pre-tokenization pattern
GPT2_PRETOK_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


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


def _load_corpus(input_path: str | os.PathLike) -> str:
    with open(input_path, "r", encoding="utf-8") as f:
        return f.read()


def _split_by_special_tokens(corpus: str, special_tokens: list[str]) -> list[str]:
    if not special_tokens:
        return [corpus]
    special_pattern = "|".join(map(re.escape, special_tokens))
    return re.split(special_pattern, corpus)


def _count_pretokens(chunks: list[str]) -> dict[str, int]:
    pretoken_counts: dict[str, int] = {}
    for chunk in chunks:
        for match in re.finditer(GPT2_PRETOK_PATTERN, chunk):
            token = match.group()
            pretoken_counts[token] = pretoken_counts.get(token, 0) + 1
    return pretoken_counts


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

    # 初始化序列
    # 将每个 pretoken 拆成初始 bytes 序列（每个元素是单字节 bytes）
    pretoken_sequences: list[tuple[list[bytes], int]] = []
    for pretoken, count in pretoken_counts.items():
        # 遍历每个字节整数 b，包装成单字节 bytes 对象
        byte_seq = [bytes([b]) for b in pretoken.encode("utf-8")]
        pretoken_sequences.append((byte_seq, count))

    # 一次性初始化 pair_freq（后续增量维护，不再重建）
    pair_freq: dict[tuple[bytes, bytes], int] = defaultdict(int)
    for seq, count in pretoken_sequences:
        for i in range(len(seq) - 1):
            pair_freq[(seq[i], seq[i + 1])] += count

    # 最大堆（负频次模拟），元素: (-freq, pair)
    # 同频时堆按 pair 字典序最小弹出（确定性即可），O(log n)
    heap: list = [_heap_entry(freq, pair) for pair, freq in pair_freq.items()]
    heapq.heapify(heap)

    # pair_freq  →  权威数据源（始终是最新频次）
    # heap       →  加速查找用的索引（可能有过期条目）
    for _ in range(merge_times):
        # 弹出最优 pair，跳过过期条目（lazy deletion）
        best_pair = None
        while heap:
            neg_freq, _, _, candidate = heapq.heappop(heap)
            if pair_freq.get(candidate, 0) == -neg_freq and -neg_freq > 0:
                best_pair = candidate
                break

        if best_pair is None:
            break

        merges.append(best_pair)
        new_token = best_pair[0] + best_pair[1]
        vocab[max(vocab) + 1] = new_token

        # 增量更新：只扫描可能包含 best_pair 的序列
        new_sequences: list[tuple[list[bytes], int]] = []
        for seq, count in pretoken_sequences:
            # 快速跳过不含左token的序列
            if best_pair[0] not in seq:
                new_sequences.append((seq, count))
                continue

            new_seq: list[bytes] = []
            i = 0
            while i < len(seq):
                if (
                    i < len(seq) - 1
                    and seq[i] == best_pair[0]
                    and seq[i + 1] == best_pair[1]
                ):
                    # 更新左邻居：(left, a) → (left, ab)
                    if new_seq:
                        left = new_seq[-1]
                        pair_freq[(left, best_pair[0])] -= count
                        pair_freq[(left, new_token)] += count
                        heapq.heappush(heap, _heap_entry(pair_freq[(left, new_token)], (left, new_token)))
                        # 当减少某个 pair 的频次时，必须推入新的记录，否则堆里只有旧的（过高的）频次。
                        # lazy deletion 会丢弃旧条目，但新的正确频次从未入堆，这个 pair 就永远消失
                        heapq.heappush(heap, _heap_entry(pair_freq[(left, best_pair[0])], (left, best_pair[0]))) 

                    # 更新右邻居：(b, right) → (ab, right)
                    if i + 2 < len(seq):
                        right = seq[i + 2]
                        pair_freq[(best_pair[1], right)] -= count
                        pair_freq[(new_token, right)] += count
                        heapq.heappush(heap, _heap_entry(pair_freq[(new_token, right)], (new_token, right)))
                        heapq.heappush(heap, _heap_entry(pair_freq[(best_pair[1], right)], (best_pair[1], right)))  

                    new_seq.append(new_token)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1

            new_sequences.append((new_seq, count))

        del pair_freq[best_pair]
        pretoken_sequences = new_sequences

    return merges

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    _validate_vocab_size(vocab_size, len(special_tokens))

    vocab = _init_base_vocab(special_tokens)
    corpus = _load_corpus(input_path)
    chunks = _split_by_special_tokens(corpus, special_tokens)
    pretoken_counts = _count_pretokens(chunks)
    merges = _compute_bpe_merges(pretoken_counts, vocab, vocab_size-len(special_tokens)-256)

    return vocab, merges


if __name__ == "__main__":
    input_path = "/Users/dami/cs336/cs336-assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt"
    vocab_size = 100000
    special_tokens = ["<|endoftext|>"]
    train_bpe(input_path, vocab_size, special_tokens)
