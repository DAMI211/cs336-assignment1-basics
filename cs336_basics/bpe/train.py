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


def _init_base_vocab() -> dict[int, bytes]:
    return {i: bytes([i]) for i in range(256)}


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


def _compute_bpe_merges(pretoken_counts: dict[str, int], vocab: dict[int, bytes]) -> list[tuple[bytes, bytes]]:
    pass


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    _validate_vocab_size(vocab_size, len(special_tokens))

    vocab = _init_base_vocab()
    corpus = _load_corpus(input_path)
    chunks = _split_by_special_tokens(corpus, special_tokens)
    pretoken_counts = _count_pretokens(chunks)
    merges = _compute_bpe_merges(pretoken_counts, vocab)

    return vocab, merges


if __name__ == "__main__":
    input_path = "/Users/dami/cs336/cs336-assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt"
    vocab_size = 100000
    special_tokens = ["<|endoftext|>"]
    train_bpe(input_path, vocab_size, special_tokens)
