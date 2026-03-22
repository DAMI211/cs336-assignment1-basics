import json
import os
import time
import functools
import psutil
import resource
from pathlib import Path
from cs336_basics.bpe.train import train_bpe

DATA_PATH = Path("/Users/dami/cs336/cs336-assignment1-basics/data")

def timer_and_memory(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        start_time = time.perf_counter()
        start_mem = process.memory_info().rss / (1024 ** 2)
        
        # 获取开关状态用于打印 (修正索引，跳过第一个参数 vocab_size)
        is_valid = kwargs.get('is_valid', args[1] if len(args) > 1 else False)
        is_owt = kwargs.get('is_owt', args[2] if len(args) > 2 else False)
        
        mode = "Validation" if is_valid else "Training"
        dataset = "OWT" if is_owt else "TinyStoriesV2"
        
        print(f"开始执行 {func.__name__} [{dataset} | {mode} 模式]...")
        print(f"初始内存使用: {start_mem:.2f} MB")
        
        result = func(*args, **kwargs)
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        # 获取峰值内存 (Peak RSS)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        if os.uname().sysname == 'Darwin':
            peak_mem = usage.ru_maxrss / (1024 ** 2)
        else:
            peak_mem = usage.ru_maxrss / 1024
            
        print(f"\n{func.__name__} [{dataset} | {mode}] 执行完成。")
        print(f"耗时: {duration:.2f} 秒 (约 {duration/60:.2f} 分钟)")
        print(f"峰值内存占用: {peak_mem:.2f} MB")
        
        return result
    return wrapper

def save_tokenizer(vocab, merges, path_prefix: Path):
    vocab_path = path_prefix.with_suffix(".vocab.json")
    merges_path = path_prefix.with_suffix(".merges.json")
    
    # 序列化词表 (bytes -> latin-1 string)
    serializable_vocab = {k: v.decode('latin-1') for k, v in vocab.items()}
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(serializable_vocab, f, indent=2, ensure_ascii=False)
        
    # 序列化合并规则 (bytes tuple -> list of strings)
    serializable_merges = [[m1.decode('latin-1'), m2.decode('latin-1')] for m1, m2 in merges]
    with open(merges_path, "w", encoding="utf-8") as f:
        json.dump(serializable_merges, f, indent=2, ensure_ascii=False)
    
    print(f"词表已保存至: {vocab_path}")
    print(f"合并规则已保存至: {merges_path}")

def load_tokenizer(path_prefix: Path):
    vocab_path = path_prefix.with_suffix(".vocab.json")
    merges_path = path_prefix.with_suffix(".merges.json")
    
    with open(vocab_path, "r", encoding="utf-8") as f:
        serializable_vocab = json.load(f)
        vocab = {int(k): v.encode('latin-1') for k, v in serializable_vocab.items()}
        
    with open(merges_path, "r", encoding="utf-8") as f:
        serializable_merges = json.load(f)
        merges = [(m[0].encode('latin-1'), m[1].encode('latin-1')) for m in serializable_merges]
                
    return vocab, merges

@timer_and_memory
def test_train_bpe(vocab_size: int, is_valid: bool = False, is_owt: bool = False):
    # 根据开关选择文件名
    data_type = "valid" if is_valid else "train"
    if is_owt:
        base_name = f"owt_{data_type}"
    else:
        base_name = f"TinyStoriesV2-GPT4-{data_type}"
        
    input_path = DATA_PATH / f"{base_name}.txt"
    output_prefix = DATA_PATH / base_name
    
    print(f"正在对 {base_name} 进行 BPE 训练，词表大小 {vocab_size:,}...")
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>"],
    )
    
    # 执行持久化
    save_tokenizer(vocab, merges, output_prefix)
    
    # 验证反序列化
    loaded_vocab, loaded_merges = load_tokenizer(output_prefix)
    assert vocab == loaded_vocab, "加载的词表与原始词表不匹配"
    assert merges == loaded_merges, "加载的合并规则与原始合并规则不匹配"
    
    # 统计最长 token
    longest_token_bytes = max(vocab.values(), key=len)
    try:
        longest_token_str = longest_token_bytes.decode('utf-8')
    except UnicodeDecodeError:
        longest_token_str = repr(longest_token_bytes)
        
    print(f"\n词表中最长的 token 长度: {len(longest_token_bytes)}")
    print(f"内容: {longest_token_str}")
    
    return vocab, merges

if __name__ == '__main__':
    # 配置开关
    # is_valid: True (处理 valid 文件), False (处理 train 文件)
    # is_owt:   True (处理 OWT 数据集), False (处理 TinyStoriesV2 数据集)
    test_train_bpe(vocab_size=10000, is_valid=True, is_owt=False)
    test_train_bpe(vocab_size=10000, is_valid=False, is_owt=False)
    test_train_bpe(vocab_size=32000, is_valid=True, is_owt=True)
    test_train_bpe(vocab_size=32000, is_valid=False, is_owt=True)
