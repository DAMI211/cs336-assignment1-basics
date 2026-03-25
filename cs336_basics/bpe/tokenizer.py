import json
import regex as re
from typing import Iterable, Iterator, Optional
from pathlib import Path

# 使用训练时一致的预分词正则
GPT2_PRETOK_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: Optional[list[str]] = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        
        # 建立反向词表，用于快速查找 Token 对应的 ID
        self.byte_to_id = {v: k for k, v in vocab.items()}
        
        # 建立特殊词到 ID 的映射
        self.special_token_to_id = {t: self.byte_to_id[t.encode("utf-8")] 
                                   for t in self.special_tokens 
                                   if t.encode("utf-8") in self.byte_to_id}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | Path,
        merges_filepath: str | Path,
        special_tokens: Optional[list[str]] = None,
    ) -> "Tokenizer":
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            serializable_vocab = json.load(f)
            vocab = {int(k): v.encode('latin-1') for k, v in serializable_vocab.items()}
            
        with open(merges_filepath, "r", encoding="utf-8") as f:
            serializable_merges = json.load(f)
            merges = [(m[0].encode('latin-1'), m[1].encode('latin-1')) for m in serializable_merges]
                    
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        if not text:
            return []

        # 1. 优先处理特殊词
        if self.special_tokens:
            # 按长度降序排序，确保匹配长词优先
            sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
            special_pattern = "(" + "|".join(re.escape(t) for t in sorted_specials) + ")"
            parts = re.split(special_pattern, text)
        else:
            parts = [text]

        ids = []
        for part in parts:
            if not part:
                continue
            
            # 如果该块是特殊词，直接获取其 ID
            if part in self.special_token_to_id:
                ids.append(self.special_token_to_id[part])
                continue
            
            # 2. 预分词：使用 finditer
            for match in re.finditer(GPT2_PRETOK_PATTERN, part):
                pt = match.group()
                # 3. 初始化：将 pre-token 转换为原始字节序列
                word = [bytes([b]) for b in pt.encode("utf-8")]
                
                # 4. 按顺序应用合并规则
                for pair in self.merges:
                    if len(word) <= 1:
                        break
                    p1, p2 = pair
                    new_word = []
                    i = 0
                    while i < len(word):
                        if i < len(word) - 1 and word[i] == p1 and word[i+1] == p2:
                            new_word.append(p1 + p2)
                            i += 2
                        else:
                            new_word.append(word[i])
                            i += 1
                    word = new_word
                
                # 5. 查表：直接映射为 ID。
                for token in word:
                    ids.append(self.byte_to_id[token])
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        byte_parts = [self.vocab[token_id] for token_id in ids]
        return b"".join(byte_parts).decode("utf-8", errors="replace")
