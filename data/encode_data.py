import numpy as np
from pathlib import Path
from cs336_basics.bpe.tokenizer import Tokenizer

# 使用相对于脚本位置的绝对路径
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR
VOCAB_DIR = SCRIPT_DIR / "vocab"   # 这里存放我们找到的 .json 文件
RESULT_DIR = SCRIPT_DIR / "result" # 这里用来存 .npy 结果

RESULT_DIR.mkdir(exist_ok=True)

def encode_dataset(tokenizer, input_path, output_path):
    print(f"Encoding {input_path}...")
    all_tokens = []
    
    # 直接以文件迭代器 f 传入 encode_iterable
    # 这会保留所有换行符并自动处理 <|endoftext|>
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        for token_id in tokenizer.encode_iterable(f):
            all_tokens.append(token_id)
    
    np_ids = np.array(all_tokens, dtype=np.uint16)
    np.save(output_path, np_ids)
    print(f"Saved to {output_path}, total tokens: {len(np_ids)}")

def main():
    # 1. 加载分词器
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

    # 2. 定义编码任务 (结果存入 RESULT_DIR)
    tasks = [
        (ts_tokenizer, DATA_DIR / "TinyStoriesV2-GPT4-train.txt", RESULT_DIR / "TinyStories_train.npy"),
        (ts_tokenizer, DATA_DIR / "TinyStoriesV2-GPT4-valid.txt", RESULT_DIR / "TinyStories_valid.npy"),
        # (owt_tokenizer, DATA_DIR / "owt_train.txt", RESULT_DIR / "owt_train.npy"),
        # (owt_tokenizer, DATA_DIR / "owt_valid.txt", RESULT_DIR / "owt_valid.npy"),
    ]

    # 3. 执行编码
    for tokenizer, input_p, output_p in tasks:
        encode_dataset(tokenizer, input_p, output_p)

if __name__ == "__main__":
    main()
