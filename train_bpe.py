import json
import os
import time
import functools
import psutil
import resource
from datetime import datetime
from pathlib import Path
from cs336_basics.bpe.train import train_bpe

DATA_PATH = Path("/Users/dami/cs336/cs336-assignment1-basics/data")

def _log(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

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
        
        _log(f"开始执行 {func.__name__} [{dataset} | {mode} 模式]...")
        _log(f"初始内存使用: {start_mem:.2f} MB")
        
        result = func(*args, **kwargs)
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        # 获取峰值内存 (Peak RSS)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        if os.uname().sysname == 'Darwin':
            peak_mem = usage.ru_maxrss / (1024 ** 2)
        else:
            peak_mem = usage.ru_maxrss / 1024
            
        _log(f"{func.__name__} [{dataset} | {mode}] 执行完成。")
        _log(f"耗时: {duration:.2f} 秒 (约 {duration/60:.2f} 分钟)")
        _log(f"峰值内存占用: {peak_mem:.2f} MB")
        
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
    
    _log(f"词表已保存至: {vocab_path}")
    _log(f"合并规则已保存至: {merges_path}")

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
def run_train_bpe(vocab_size: int, is_valid: bool = False, is_owt: bool = False):
    # 根据开关选择文件名
    data_type = "valid" if is_valid else "train"
    if is_owt:
        base_name = f"owt_{data_type}"
    else:
        base_name = f"TinyStoriesV2-GPT4-{data_type}"

    input_path = DATA_PATH / f"{base_name}.txt"
    output_prefix = DATA_PATH / base_name

    _log(f"正在对 {base_name} 进行 BPE 训练，词表大小 {vocab_size:,}...")
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
        
    _log(f"词表中最长的 token 长度: {len(longest_token_bytes)}")
    _log(f"内容: {longest_token_str}")
    
    return vocab, merges


if __name__ == '__main__':
    # 配置开关
    # is_valid: True (处理 valid 文件), False (处理 train 文件)
    # is_owt:   True (处理 OWT 数据集), False (处理 TinyStoriesV2 数据集)
    # run_train_bpe(vocab_size=10000, is_valid=True, is_owt=False)
    # run_train_bpe(vocab_size=10000, is_valid=False, is_owt=False)
    # run_train_bpe(vocab_size=32000, is_valid=True, is_owt=True)
    run_train_bpe(vocab_size=32000, is_valid=False, is_owt=True)

# -------------------------------------------------------------------------------------------------------------------------
# (cs336) dami@dami-3 cs336-assignment1-basics % python3 train_bpe.py
# [2026-03-25 23:29:09] 开始执行 run_train_bpe [TinyStoriesV2 | Training 模式]...
# [2026-03-25 23:29:09] 初始内存使用: 26.95 MB
# [2026-03-25 23:29:09] 正在对 TinyStoriesV2-GPT4-train 进行 BPE 训练，词表大小 10,000...
# [2026-03-25 23:29:09] Vocab size validated.
# [2026-03-25 23:29:09] Base vocab initialized with 257 tokens.
# [2026-03-25 23:29:09] Chunk boundaries found. Split file into 10 chunks.
# [2026-03-25 23:29:32] Parallel counting of tokens completed.
# [2026-03-25 23:29:32] Aggregated pretoken counts. Found 59933 unique pretokens.
# [2026-03-25 23:29:32] Starting BPE merges (total: 9743)...
# [2026-03-25 23:29:33] Merge 100/9743: best_pair=(b' ha', b'pp'), freq=3147884
# [2026-03-25 23:29:33] Merge 200/9743: best_pair=(b' s', b'e'), freq=1410130
# [2026-03-25 23:29:33] Merge 300/9743: best_pair=(b' s', b'omet'), freq=790510
# [2026-03-25 23:29:33] Merge 400/9743: best_pair=(b' g', b'ot'), freq=524776
# [2026-03-25 23:29:33] Merge 500/9743: best_pair=(b' e', b'ach'), freq=369637
# [2026-03-25 23:29:33] Merge 600/9743: best_pair=(b'l', b'f'), freq=279566
# [2026-03-25 23:29:33] Merge 700/9743: best_pair=(b' wal', b'k'), freq=221114
# [2026-03-25 23:29:33] Merge 800/9743: best_pair=(b' do', b'll'), freq=177602
# [2026-03-25 23:29:33] Merge 900/9743: best_pair=(b' ', b'G'), freq=147699
# [2026-03-25 23:29:33] Merge 1000/9743: best_pair=(b'ec', b't'), freq=127288
# [2026-03-25 23:29:33] Merge 1100/9743: best_pair=(b' l', b'ight'), freq=108006
# [2026-03-25 23:29:33] Merge 1200/9743: best_pair=(b' d', b'in'), freq=92211
# [2026-03-25 23:29:33] Merge 1300/9743: best_pair=(b' picture', b's'), freq=80416
# [2026-03-25 23:29:34] Merge 1400/9743: best_pair=(b'itt', b'en'), freq=68466
# [2026-03-25 23:29:34] Merge 1500/9743: best_pair=(b'A', b'my'), freq=59829
# [2026-03-25 23:29:34] Merge 1600/9743: best_pair=(b' tal', b'king'), freq=53781
# [2026-03-25 23:29:34] Merge 1700/9743: best_pair=(b'b', b'all'), freq=48005
# [2026-03-25 23:29:34] Merge 1800/9743: best_pair=(b' k', b'iss'), freq=43477
# [2026-03-25 23:29:34] Merge 1900/9743: best_pair=(b' str', b'ing'), freq=38985
# [2026-03-25 23:29:34] Merge 2000/9743: best_pair=(b' adventure', b's'), freq=35431
# [2026-03-25 23:29:34] Merge 2100/9743: best_pair=(b' spr', b'ay'), freq=32681
# [2026-03-25 23:29:34] Merge 2200/9743: best_pair=(b' prin', b'ce'), freq=30205
# [2026-03-25 23:29:34] Merge 2300/9743: best_pair=(b' we', b'aring'), freq=27622
# [2026-03-25 23:29:34] Merge 2400/9743: best_pair=(b' f', b'at'), freq=25325
# [2026-03-25 23:29:34] Merge 2500/9743: best_pair=(b' j', b'am'), freq=23459
# [2026-03-25 23:29:34] Merge 2600/9743: best_pair=(b'o', b'nd'), freq=21888
# [2026-03-25 23:29:34] Merge 2700/9743: best_pair=(b' adventur', b'ous'), freq=20513
# [2026-03-25 23:29:34] Merge 2800/9743: best_pair=(b' medic', b'ine'), freq=19443
# [2026-03-25 23:29:34] Merge 2900/9743: best_pair=(b' cheer', b'ful'), freq=18382
# [2026-03-25 23:29:34] Merge 3000/9743: best_pair=(b'il', b'ty'), freq=17394
# [2026-03-25 23:29:34] Merge 3100/9743: best_pair=(b' p', b'ipe'), freq=16615
# [2026-03-25 23:29:34] Merge 3200/9743: best_pair=(b'as', b'ing'), freq=15875
# [2026-03-25 23:29:34] Merge 3300/9743: best_pair=(b' coo', b'ked'), freq=15112
# [2026-03-25 23:29:34] Merge 3400/9743: best_pair=(b'G', b'ive'), freq=14418
# [2026-03-25 23:29:34] Merge 3500/9743: best_pair=(b' read', b'ing'), freq=13579
# [2026-03-25 23:29:34] Merge 3600/9743: best_pair=(b' r', b'ot'), freq=13017
# [2026-03-25 23:29:34] Merge 3700/9743: best_pair=(b' batter', b'y'), freq=12454
# [2026-03-25 23:29:34] Merge 3800/9743: best_pair=(b' o', b'ak'), freq=11945
# [2026-03-25 23:29:34] Merge 3900/9743: best_pair=(b' stop', b's'), freq=11337
# [2026-03-25 23:29:34] Merge 4000/9743: best_pair=(b' ex', b't'), freq=10797
# [2026-03-25 23:29:34] Merge 4100/9743: best_pair=(b'ang', b'u'), freq=10132
# [2026-03-25 23:29:34] Merge 4200/9743: best_pair=(b' bright', b'ly'), freq=9465
# [2026-03-25 23:29:34] Merge 4300/9743: best_pair=(b' B', b'ill'), freq=8923
# [2026-03-25 23:29:34] Merge 4400/9743: best_pair=(b' se', b'cond'), freq=8306
# [2026-03-25 23:29:34] Merge 4500/9743: best_pair=(b'M', b'ark'), freq=7626
# [2026-03-25 23:29:34] Merge 4600/9743: best_pair=(b' blow', b'ing'), freq=7113
# [2026-03-25 23:29:34] Merge 4700/9743: best_pair=(b' com', b'f'), freq=6603
# [2026-03-25 23:29:34] Merge 4800/9743: best_pair=(b' snugg', b'led'), freq=6164
# [2026-03-25 23:29:34] Merge 4900/9743: best_pair=(b' E', b'ventually'), freq=5738
# [2026-03-25 23:29:34] Merge 5000/9743: best_pair=(b' vis', b'ited'), freq=5351
# [2026-03-25 23:29:34] Merge 5100/9743: best_pair=(b'un', b'e'), freq=4948
# [2026-03-25 23:29:34] Merge 5200/9743: best_pair=(b' str', b'ang'), freq=4554
# [2026-03-25 23:29:34] Merge 5300/9743: best_pair=(b' so', b're'), freq=4226
# [2026-03-25 23:29:34] Merge 5400/9743: best_pair=(b'pl', b'ash'), freq=3885
# [2026-03-25 23:29:34] Merge 5500/9743: best_pair=(b' own', b'ed'), freq=3589
# [2026-03-25 23:29:34] Merge 5600/9743: best_pair=(b'is', b'dom'), freq=3307
# [2026-03-25 23:29:34] Merge 5700/9743: best_pair=(b'R', b'o'), freq=3119
# [2026-03-25 23:29:34] Merge 5800/9743: best_pair=(b' be', b'ad'), freq=2930
# [2026-03-25 23:29:34] Merge 5900/9743: best_pair=(b' firef', b'ight'), freq=2712
# [2026-03-25 23:29:34] Merge 6000/9743: best_pair=(b' C', b'uddles'), freq=2542
# [2026-03-25 23:29:34] Merge 6100/9743: best_pair=(b'F', b'ine'), freq=2395
# [2026-03-25 23:29:34] Merge 6200/9743: best_pair=(b' K', b'ids'), freq=2241
# [2026-03-25 23:29:34] Merge 6300/9743: best_pair=(b' spe', b'aking'), freq=2101
# [2026-03-25 23:29:34] Merge 6400/9743: best_pair=(b' chef', b's'), freq=1996
# [2026-03-25 23:29:34] Merge 6500/9743: best_pair=(b' m', b'ater'), freq=1881
# [2026-03-25 23:29:34] Merge 6600/9743: best_pair=(b' forget', b's'), freq=1790
# [2026-03-25 23:29:34] Merge 6700/9743: best_pair=(b'r', b'ig'), freq=1683
# [2026-03-25 23:29:34] Merge 6800/9743: best_pair=(b' snow', b'fl'), freq=1599
# [2026-03-25 23:29:34] Merge 6900/9743: best_pair=(b' sing', b'le'), freq=1521
# [2026-03-25 23:29:34] Merge 7000/9743: best_pair=(b' hor', b'iz'), freq=1438
# [2026-03-25 23:29:34] Merge 7100/9743: best_pair=(b' fl', b'ipped'), freq=1366
# [2026-03-25 23:29:34] Merge 7200/9743: best_pair=(b' bit', b's'), freq=1289
# [2026-03-25 23:29:34] Merge 7300/9743: best_pair=(b' magaz', b'ines'), freq=1222
# [2026-03-25 23:29:34] Merge 7400/9743: best_pair=(b'Sn', b'ow'), freq=1157
# [2026-03-25 23:29:34] Merge 7500/9743: best_pair=(b' encourage', b'ment'), freq=1095
# [2026-03-25 23:29:34] Merge 7600/9743: best_pair=(b'C', b'arl'), freq=1048
# [2026-03-25 23:29:34] Merge 7700/9743: best_pair=(b' sweat', b'ers'), freq=1003
# [2026-03-25 23:29:34] Merge 7800/9743: best_pair=(b' bother', b'ed'), freq=969
# [2026-03-25 23:29:34] Merge 7900/9743: best_pair=(b' H', b'oney'), freq=919
# [2026-03-25 23:29:34] Merge 8000/9743: best_pair=(b' mom', b'mies'), freq=879
# [2026-03-25 23:29:34] Merge 8100/9743: best_pair=(b' cryst', b'als'), freq=840
# [2026-03-25 23:29:34] Merge 8200/9743: best_pair=(b'm', b'ine'), freq=809
# [2026-03-25 23:29:34] Merge 8300/9743: best_pair=(b' support', b'ing'), freq=778
# [2026-03-25 23:29:35] Merge 8400/9743: best_pair=(b' neare', b'st'), freq=747
# [2026-03-25 23:29:35] Merge 8500/9743: best_pair=(b' argum', b'ent'), freq=716
# [2026-03-25 23:29:35] Merge 8600/9743: best_pair=(b' express', b'ion'), freq=691
# [2026-03-25 23:29:35] Merge 8700/9743: best_pair=(b'ri', b'ck'), freq=660
# [2026-03-25 23:29:35] Merge 8800/9743: best_pair=(b' cop', b's'), freq=633
# [2026-03-25 23:29:35] Merge 8900/9743: best_pair=(b' Gra', b'nd'), freq=609
# [2026-03-25 23:29:35] Merge 9000/9743: best_pair=(b'ch', b'ing'), freq=584
# [2026-03-25 23:29:35] Merge 9100/9743: best_pair=(b'amil', b'iar'), freq=561
# [2026-03-25 23:29:35] Merge 9200/9743: best_pair=(b'C', b'he'), freq=542
# [2026-03-25 23:29:35] Merge 9300/9743: best_pair=(b' la', b'va'), freq=521
# [2026-03-25 23:29:35] Merge 9400/9743: best_pair=(b' p', b'ounced'), freq=502
# [2026-03-25 23:29:35] Merge 9500/9743: best_pair=(b'it', b'es'), freq=485
# [2026-03-25 23:29:35] Merge 9600/9743: best_pair=(b' qu', b'ilt'), freq=469
# [2026-03-25 23:29:35] Merge 9700/9743: best_pair=(b'solut', b'ely'), freq=454
# [2026-03-25 23:29:35] BPE merges completed.
# [2026-03-25 23:29:35] 词表已保存至: /Users/dami/cs336/cs336-assignment1-basics/data/TinyStoriesV2-GPT4-train.vocab.json
# [2026-03-25 23:29:35] 合并规则已保存至: /Users/dami/cs336/cs336-assignment1-basics/data/TinyStoriesV2-GPT4-train.merges.json
# [2026-03-25 23:29:35] 词表中最长的 token 长度: 15
# [2026-03-25 23:29:35] 内容:  accomplishment
# [2026-03-25 23:29:35] run_train_bpe [TinyStoriesV2 | Training] 执行完成。
# [2026-03-25 23:29:35] 耗时: 26.05 秒 (约 0.43 分钟)
# [2026-03-25 23:29:35] 峰值内存占用: 160.52 MB
# -------------------------------------------------------------------------------------------------------------------------
# (cs336) dami@dami-3 cs336-assignment1-basics % python3 train_bpe.py
# [2026-03-25 23:30:26] 开始执行 run_train_bpe [OWT | Training 模式]...
# [2026-03-25 23:30:26] 初始内存使用: 27.14 MB
# [2026-03-25 23:30:26] 正在对 owt_train 进行 BPE 训练，词表大小 32,000...
# [2026-03-25 23:30:26] Vocab size validated.
# [2026-03-25 23:30:26] Base vocab initialized with 257 tokens.
# [2026-03-25 23:30:26] Chunk boundaries found. Split file into 10 chunks.
# [2026-03-25 23:34:05] Parallel counting of tokens completed.
# [2026-03-25 23:34:12] Aggregated pretoken counts. Found 6601892 unique pretokens.
# [2026-03-25 23:34:12] Starting BPE merges (total: 31743)...
# [2026-03-25 23:36:22] Merge 100/31743: best_pair=(b'i', b'll'), freq=13619361
# [2026-03-25 23:37:05] Merge 200/31743: best_pair=(b' p', b'l'), freq=6703800
# [2026-03-25 23:37:23] Merge 300/31743: best_pair=(b' T', b'h'), freq=3922372
# [2026-03-25 23:37:37] Merge 400/31743: best_pair=(b'or', b'y'), freq=2784423
# [2026-03-25 23:37:50] Merge 500/31743: best_pair=(b'e', b'c'), freq=1976873
# [2026-03-25 23:38:01] Merge 600/31743: best_pair=(b't', b'le'), freq=1600663
# [2026-03-25 23:38:05] Merge 700/31743: best_pair=(b'a', b'x'), freq=1322648
# [2026-03-25 23:38:08] Merge 800/31743: best_pair=(b'in', b's'), freq=1142167
# [2026-03-25 23:38:19] Merge 900/31743: best_pair=(b'an', b'c'), freq=985177
# [2026-03-25 23:38:21] Merge 1000/31743: best_pair=(b' R', b'ep'), freq=870461
# [2026-03-25 23:38:23] Merge 1100/31743: best_pair=(b't', b'o'), freq=766796
# [2026-03-25 23:38:25] Merge 1200/31743: best_pair=(b'er', b'tain'), freq=685676
# [2026-03-25 23:38:36] Merge 1300/31743: best_pair=(b'al', b'f'), freq=625628
# [2026-03-25 23:38:38] Merge 1400/31743: best_pair=(b'at', b's'), freq=575006
# [2026-03-25 23:38:39] Merge 1500/31743: best_pair=(b' con', b'cer'), freq=533597
# [2026-03-25 23:38:41] Merge 1600/31743: best_pair=(b'.', b','), freq=489232
# [2026-03-25 23:38:42] Merge 1700/31743: best_pair=(b'ra', b'el'), freq=455628
# [2026-03-25 23:38:44] Merge 1800/31743: best_pair=(b' M', b'an'), freq=426984
# [2026-03-25 23:38:55] Merge 1900/31743: best_pair=(b' ch', b'all'), freq=399668
# [2026-03-25 23:38:57] Merge 2000/31743: best_pair=(b' ab', b'ove'), freq=380818
# [2026-03-25 23:38:58] Merge 2100/31743: best_pair=(b' includ', b'e'), freq=357875
# [2026-03-25 23:38:59] Merge 2200/31743: best_pair=(b' dec', b'l'), freq=335839
# [2026-03-25 23:39:00] Merge 2300/31743: best_pair=(b' ear', b'lier'), freq=319350
# [2026-03-25 23:39:02] Merge 2400/31743: best_pair=(b' ne', b'arly'), freq=303109
# [2026-03-25 23:39:04] Merge 2500/31743: best_pair=(b'r', b'ain'), freq=288727
# [2026-03-25 23:39:05] Merge 2600/31743: best_pair=(b' G', b'l'), freq=274167
# [2026-03-25 23:39:06] Merge 2700/31743: best_pair=(b' Brit', b'ish'), freq=262125
# [2026-03-25 23:39:07] Merge 2800/31743: best_pair=(b' lead', b'er'), freq=249779
# [2026-03-25 23:39:19] Merge 2900/31743: best_pair=(b' see', b'k'), freq=237303
# [2026-03-25 23:39:20] Merge 3000/31743: best_pair=(b' offic', b'ers'), freq=227154
# [2026-03-25 23:39:21] Merge 3100/31743: best_pair=(b'y', b'l'), freq=218404
# [2026-03-25 23:39:22] Merge 3200/31743: best_pair=(b' c', b'at'), freq=208887
# [2026-03-25 23:39:24] Merge 3300/31743: best_pair=(b'O', b'T'), freq=201006
# [2026-03-25 23:39:25] Merge 3400/31743: best_pair=(b' book', b's'), freq=192666
# [2026-03-25 23:39:26] Merge 3500/31743: best_pair=(b' sen', b'ior'), freq=185113
# [2026-03-25 23:39:27] Merge 3600/31743: best_pair=(b' larg', b'er'), freq=177747
# [2026-03-25 23:39:28] Merge 3700/31743: best_pair=(b' ca', b'used'), freq=172039
# [2026-03-25 23:39:30] Merge 3800/31743: best_pair=(b' l', b'ik'), freq=165231
# [2026-03-25 23:39:31] Merge 3900/31743: best_pair=(b' wh', b'atever'), freq=158570
# [2026-03-25 23:39:32] Merge 4000/31743: best_pair=(b'ord', b's'), freq=152571
# [2026-03-25 23:39:33] Merge 4100/31743: best_pair=(b' understand', b'ing'), freq=147712
# [2026-03-25 23:39:34] Merge 4200/31743: best_pair=(b' Inst', b'itute'), freq=143433
# [2026-03-25 23:39:35] Merge 4300/31743: best_pair=(b' st', b'ated'), freq=139096
# [2026-03-25 23:39:48] Merge 4400/31743: best_pair=(b'az', b'on'), freq=134881
# [2026-03-25 23:39:50] Merge 4500/31743: best_pair=(b' tell', b's'), freq=131068
# [2026-03-25 23:39:51] Merge 4600/31743: best_pair=(b' girl', b's'), freq=127237
# [2026-03-25 23:39:52] Merge 4700/31743: best_pair=(b'"', b':'), freq=123119
# [2026-03-25 23:39:53] Merge 4800/31743: best_pair=(b' car', b'ried'), freq=119689
# [2026-03-25 23:39:54] Merge 4900/31743: best_pair=(b'v', b'is'), freq=116044
# [2026-03-25 23:39:55] Merge 5000/31743: best_pair=(b' Tor', b'onto'), freq=112475
# [2026-03-25 23:39:56] Merge 5100/31743: best_pair=(b'I', b'ON'), freq=108779
# [2026-03-25 23:39:57] Merge 5200/31743: best_pair=(b'n', b'oon'), freq=105726
# [2026-03-25 23:39:58] Merge 5300/31743: best_pair=(b' t', b'act'), freq=102590
# [2026-03-25 23:39:59] Merge 5400/31743: best_pair=(b' pan', b'el'), freq=100290
# [2026-03-25 23:39:59] Merge 5500/31743: best_pair=(b'U', b'T'), freq=97916
# [2026-03-25 23:40:00] Merge 5600/31743: best_pair=(b'ok', b'en'), freq=95326
# [2026-03-25 23:40:01] Merge 5700/31743: best_pair=(b'o', b'oth'), freq=93144
# [2026-03-25 23:40:02] Merge 5800/31743: best_pair=(b'g', b'ia'), freq=90977
# [2026-03-25 23:40:03] Merge 5900/31743: best_pair=(b' mag', b'azine'), freq=89058
# [2026-03-25 23:40:04] Merge 6000/31743: best_pair=(b' consist', b'ent'), freq=87177
# [2026-03-25 23:40:04] Merge 6100/31743: best_pair=(b' bu', b'ying'), freq=85306
# [2026-03-25 23:40:05] Merge 6200/31743: best_pair=(b' Y', b'ear'), freq=83427
# [2026-03-25 23:40:06] Merge 6300/31743: best_pair=(b'ke', b'l'), freq=81600
# [2026-03-25 23:40:07] Merge 6400/31743: best_pair=(b' j', b'ourney'), freq=79905
# [2026-03-25 23:40:21] Merge 6500/31743: best_pair=(b'az', b'e'), freq=78349
# [2026-03-25 23:40:22] Merge 6600/31743: best_pair=(b'ob', b'ile'), freq=76702
# [2026-03-25 23:40:23] Merge 6700/31743: best_pair=(b' po', b'verty'), freq=75053
# [2026-03-25 23:40:23] Merge 6800/31743: best_pair=(b'ort', b'ed'), freq=73446
# [2026-03-25 23:40:24] Merge 6900/31743: best_pair=(b' se', b'ats'), freq=71832
# [2026-03-25 23:40:25] Merge 7000/31743: best_pair=(b' cont', b'em'), freq=70308
# [2026-03-25 23:40:25] Merge 7100/31743: best_pair=(b' inn', b'oc'), freq=68985
# [2026-03-25 23:40:26] Merge 7200/31743: best_pair=(b' scen', b'es'), freq=67429
# [2026-03-25 23:40:27] Merge 7300/31743: best_pair=(b' vis', b'ited'), freq=66142
# [2026-03-25 23:40:27] Merge 7400/31743: best_pair=(b' tax', b'p'), freq=64935
# [2026-03-25 23:40:28] Merge 7500/31743: best_pair=(b' scen', b'ario'), freq=63604
# [2026-03-25 23:40:29] Merge 7600/31743: best_pair=(b' cann', b'abis'), freq=62377
# [2026-03-25 23:40:30] Merge 7700/31743: best_pair=(b' F', b'red'), freq=61062
# [2026-03-25 23:40:30] Merge 7800/31743: best_pair=(b' Per', b'haps'), freq=59633
# [2026-03-25 23:40:31] Merge 7900/31743: best_pair=(b' A', b'BC'), freq=58357
# [2026-03-25 23:40:32] Merge 8000/31743: best_pair=(b' ESP', b'N'), freq=57362
# [2026-03-25 23:40:32] Merge 8100/31743: best_pair=(b' g', b'athered'), freq=56271
# [2026-03-25 23:40:33] Merge 8200/31743: best_pair=(b'ition', b'ally'), freq=55338
# [2026-03-25 23:40:33] Merge 8300/31743: best_pair=(b' mon', b'itor'), freq=54304
# [2026-03-25 23:40:34] Merge 8400/31743: best_pair=(b' Sh', b'are'), freq=53552
# [2026-03-25 23:40:34] Merge 8500/31743: best_pair=(b' l', b'ucky'), freq=52509
# [2026-03-25 23:40:35] Merge 8600/31743: best_pair=(b' respons', b'es'), freq=51436
# [2026-03-25 23:40:35] Merge 8700/31743: best_pair=(b' cl', b'ock'), freq=50456
# [2026-03-25 23:40:36] Merge 8800/31743: best_pair=(b'ag', b'ic'), freq=49698
# [2026-03-25 23:40:37] Merge 8900/31743: best_pair=(b' equ', b'ality'), freq=48831
# [2026-03-25 23:40:37] Merge 9000/31743: best_pair=(b'ord', b'inary'), freq=47979
# [2026-03-25 23:40:38] Merge 9100/31743: best_pair=(b' question', b'ed'), freq=47275
# [2026-03-25 23:40:39] Merge 9200/31743: best_pair=(b'a', b'a'), freq=46623
# [2026-03-25 23:40:39] Merge 9300/31743: best_pair=(b' T', b'en'), freq=45931
# [2026-03-25 23:40:40] Merge 9400/31743: best_pair=(b'y', b'es'), freq=45237
# [2026-03-25 23:40:40] Merge 9500/31743: best_pair=(b' H', b'old'), freq=44560
# [2026-03-25 23:40:41] Merge 9600/31743: best_pair=(b' Mar', b'x'), freq=43899
# [2026-03-25 23:40:41] Merge 9700/31743: best_pair=(b' hosp', b'itals'), freq=43178
# [2026-03-25 23:40:42] Merge 9800/31743: best_pair=(b' egg', b's'), freq=42425
# [2026-03-25 23:40:42] Merge 9900/31743: best_pair=(b' gen', b're'), freq=41770
# [2026-03-25 23:40:58] Merge 10000/31743: best_pair=(b' gl', b'ad'), freq=41062
# [2026-03-25 23:40:58] Merge 10100/31743: best_pair=(b'ro', b't'), freq=40385
# [2026-03-25 23:40:59] Merge 10200/31743: best_pair=(b'ial', b's'), freq=39807
# [2026-03-25 23:41:00] Merge 10300/31743: best_pair=(b' S', b'H'), freq=39144
# [2026-03-25 23:41:00] Merge 10400/31743: best_pair=(b' st', b'ating'), freq=38579
# [2026-03-25 23:41:01] Merge 10500/31743: best_pair=(b' Sher', b'iff'), freq=37983
# [2026-03-25 23:41:01] Merge 10600/31743: best_pair=(b'oph', b'ob'), freq=37454
# [2026-03-25 23:41:02] Merge 10700/31743: best_pair=(b' ev', b'olved'), freq=36843
# [2026-03-25 23:41:02] Merge 10800/31743: best_pair=(b'look', b'ing'), freq=36449
# [2026-03-25 23:41:03] Merge 10900/31743: best_pair=(b' 6', b'3'), freq=35960
# [2026-03-25 23:41:03] Merge 11000/31743: best_pair=(b' su', b'ck'), freq=35473
# [2026-03-25 23:41:04] Merge 11100/31743: best_pair=(b'm', b'ons'), freq=34980
# [2026-03-25 23:41:04] Merge 11200/31743: best_pair=(b' fuck', b'ing'), freq=34534
# [2026-03-25 23:41:05] Merge 11300/31743: best_pair=(b' co', b'in'), freq=34028
# [2026-03-25 23:41:05] Merge 11400/31743: best_pair=(b' restrict', b'ed'), freq=33572
# [2026-03-25 23:41:06] Merge 11500/31743: best_pair=(b' M', b'and'), freq=33078
# [2026-03-25 23:41:06] Merge 11600/31743: best_pair=(b'od', b'a'), freq=32647
# [2026-03-25 23:41:06] Merge 11700/31743: best_pair=(b' gra', b've'), freq=32231
# [2026-03-25 23:41:07] Merge 11800/31743: best_pair=(b'appropri', b'ate'), freq=31868
# [2026-03-25 23:41:07] Merge 11900/31743: best_pair=(b' aw', b'kward'), freq=31422
# [2026-03-25 23:41:08] Merge 12000/31743: best_pair=(b' dec', b'or'), freq=31014
# [2026-03-25 23:41:08] Merge 12100/31743: best_pair=(b' Net', b'anyahu'), freq=30657
# [2026-03-25 23:41:09] Merge 12200/31743: best_pair=(b' space', b'craft'), freq=30291
# [2026-03-25 23:41:09] Merge 12300/31743: best_pair=(b' t', b'el'), freq=29861
# [2026-03-25 23:41:10] Merge 12400/31743: best_pair=(b'Wh', b'ich'), freq=29465
# [2026-03-25 23:41:10] Merge 12500/31743: best_pair=(b' Ant', b'i'), freq=29061
# [2026-03-25 23:41:11] Merge 12600/31743: best_pair=(b'l', b'as'), freq=28668
# [2026-03-25 23:41:11] Merge 12700/31743: best_pair=(b' tradition', b'ally'), freq=28370
# [2026-03-25 23:41:12] Merge 12800/31743: best_pair=(b'B', b'et'), freq=28025
# [2026-03-25 23:41:13] Merge 12900/31743: best_pair=(b' G', b'host'), freq=27653
# [2026-03-25 23:41:13] Merge 13000/31743: best_pair=(b' st', b'abil'), freq=27344
# [2026-03-25 23:41:14] Merge 13100/31743: best_pair=(b' P', b'interest'), freq=27047
# [2026-03-25 23:41:14] Merge 13200/31743: best_pair=(b' tem', b'plate'), freq=26719
# [2026-03-25 23:41:15] Merge 13300/31743: best_pair=(b' L', b'ED'), freq=26432
# [2026-03-25 23:41:15] Merge 13400/31743: best_pair=(b' endors', b'ed'), freq=26143
# [2026-03-25 23:41:16] Merge 13500/31743: best_pair=(b' consum', b'ed'), freq=25833
# [2026-03-25 23:41:16] Merge 13600/31743: best_pair=(b' special', b'ist'), freq=25533
# [2026-03-25 23:41:17] Merge 13700/31743: best_pair=(b'ili', b'ation'), freq=25226
# [2026-03-25 23:41:17] Merge 13800/31743: best_pair=(b'C', b'OM'), freq=24937
# [2026-03-25 23:41:18] Merge 13900/31743: best_pair=(b' or', b'che'), freq=24670
# [2026-03-25 23:41:18] Merge 14000/31743: best_pair=(b'p', b'rom'), freq=24418
# [2026-03-25 23:41:19] Merge 14100/31743: best_pair=(b' class', b'room'), freq=24199
# [2026-03-25 23:41:19] Merge 14200/31743: best_pair=(b'ur', b'ring'), freq=23962
# [2026-03-25 23:41:20] Merge 14300/31743: best_pair=(b' g', b'rey'), freq=23721
# [2026-03-25 23:41:20] Merge 14400/31743: best_pair=(b' out', b'lined'), freq=23421
# [2026-03-25 23:41:21] Merge 14500/31743: best_pair=(b'm', b'ile'), freq=23164
# [2026-03-25 23:41:21] Merge 14600/31743: best_pair=(b'uster', b'ity'), freq=22950
# [2026-03-25 23:41:21] Merge 14700/31743: best_pair=(b'eg', b'u'), freq=22687
# [2026-03-25 23:41:22] Merge 14800/31743: best_pair=(b'atic', b'an'), freq=22425
# [2026-03-25 23:41:22] Merge 14900/31743: best_pair=(b' r', b'ats'), freq=22196
# [2026-03-25 23:41:23] Merge 15000/31743: best_pair=(b' revel', b'ation'), freq=22003
# [2026-03-25 23:41:23] Merge 15100/31743: best_pair=(b' B', b'L'), freq=21780
# [2026-03-25 23:41:24] Merge 15200/31743: best_pair=(b' B', b'annon'), freq=21577
# [2026-03-25 23:41:24] Merge 15300/31743: best_pair=(b' b', b'out'), freq=21389
# [2026-03-25 23:41:25] Merge 15400/31743: best_pair=(b' stick', b'ing'), freq=21148
# [2026-03-25 23:41:25] Merge 15500/31743: best_pair=(b' Al', b'b'), freq=20926
# [2026-03-25 23:41:43] Merge 15600/31743: best_pair=(b' design', b'ing'), freq=20704
# [2026-03-25 23:41:43] Merge 15700/31743: best_pair=(b' n', b'arc'), freq=20448
# [2026-03-25 23:41:44] Merge 15800/31743: best_pair=(b' tu', b'ition'), freq=20260
# [2026-03-25 23:41:44] Merge 15900/31743: best_pair=(b' laun', b'ches'), freq=20062
# [2026-03-25 23:41:45] Merge 16000/31743: best_pair=(b' pict', b'ured'), freq=19833
# [2026-03-25 23:41:45] Merge 16100/31743: best_pair=(b' O', b'z'), freq=19646
# [2026-03-25 23:41:46] Merge 16200/31743: best_pair=(b' sub', b't'), freq=19466
# [2026-03-25 23:41:46] Merge 16300/31743: best_pair=(b'>', b'<'), freq=19282
# [2026-03-25 23:41:47] Merge 16400/31743: best_pair=(b'ues', b'e'), freq=19100
# [2026-03-25 23:41:47] Merge 16500/31743: best_pair=(b' vo', b'iced'), freq=18867
# [2026-03-25 23:41:48] Merge 16600/31743: best_pair=(b' finger', b'print'), freq=18679
# [2026-03-25 23:41:48] Merge 16700/31743: best_pair=(b' S', b'au'), freq=18491
# [2026-03-25 23:41:48] Merge 16800/31743: best_pair=(b't', b'i'), freq=18284
# [2026-03-25 23:41:49] Merge 16900/31743: best_pair=(b'A', b'E'), freq=18113
# [2026-03-25 23:41:49] Merge 17000/31743: best_pair=(b'on', b'gh'), freq=17909
# [2026-03-25 23:41:50] Merge 17100/31743: best_pair=(b' impact', b'ed'), freq=17749
# [2026-03-25 23:41:50] Merge 17200/31743: best_pair=(b' des', b'erved'), freq=17580
# [2026-03-25 23:41:50] Merge 17300/31743: best_pair=(b' Gener', b'ation'), freq=17408
# [2026-03-25 23:41:51] Merge 17400/31743: best_pair=(b' F', b'ix'), freq=17246
# [2026-03-25 23:41:51] Merge 17500/31743: best_pair=(b' S', b'add'), freq=17111
# [2026-03-25 23:41:52] Merge 17600/31743: best_pair=(b'add', b'afi'), freq=16958
# [2026-03-25 23:41:52] Merge 17700/31743: best_pair=(b' upgr', b'aded'), freq=16787
# [2026-03-25 23:41:53] Merge 17800/31743: best_pair=(b' dep', b'ressed'), freq=16640
# [2026-03-25 23:41:53] Merge 17900/31743: best_pair=(b' interf', b'ere'), freq=16494
# [2026-03-25 23:41:53] Merge 18000/31743: best_pair=(b'AT', b'H'), freq=16359
# [2026-03-25 23:41:54] Merge 18100/31743: best_pair=(b' pl', b'um'), freq=16243
# [2026-03-25 23:41:54] Merge 18200/31743: best_pair=(b' wond', b'ers'), freq=16121
# [2026-03-25 23:41:55] Merge 18300/31743: best_pair=(b'iz', b'oph'), freq=16010
# [2026-03-25 23:41:55] Merge 18400/31743: best_pair=(b'es', b'ar'), freq=15882
# [2026-03-25 23:41:55] Merge 18500/31743: best_pair=(b' promot', b'es'), freq=15717
# [2026-03-25 23:41:56] Merge 18600/31743: best_pair=(b' R', b'B'), freq=15574
# [2026-03-25 23:41:56] Merge 18700/31743: best_pair=(b' S', b'ask'), freq=15436
# [2026-03-25 23:41:57] Merge 18800/31743: best_pair=(b'oint', b'ed'), freq=15267
# [2026-03-25 23:41:58] Merge 18900/31743: best_pair=(b'op', b'ath'), freq=15143
# [2026-03-25 23:41:58] Merge 19000/31743: best_pair=(b' Inn', b'ov'), freq=15014
# [2026-03-25 23:41:59] Merge 19100/31743: best_pair=(b' method', b'ology'), freq=14889
# [2026-03-25 23:41:59] Merge 19200/31743: best_pair=(b' l', b'ining'), freq=14764
# [2026-03-25 23:41:59] Merge 19300/31743: best_pair=(b' Be', b'gin'), freq=14625
# [2026-03-25 23:42:00] Merge 19400/31743: best_pair=(b' num', b'er'), freq=14504
# [2026-03-25 23:42:00] Merge 19500/31743: best_pair=(b' rest', b'art'), freq=14361
# [2026-03-25 23:42:01] Merge 19600/31743: best_pair=(b' cap', b'ita'), freq=14247
# [2026-03-25 23:42:02] Merge 19700/31743: best_pair=(b'sequ', b'ently'), freq=14130
# [2026-03-25 23:42:02] Merge 19800/31743: best_pair=(b'ist', b'ical'), freq=14016
# [2026-03-25 23:42:02] Merge 19900/31743: best_pair=(b'12', b'3'), freq=13918
# [2026-03-25 23:42:03] Merge 20000/31743: best_pair=(b' Person', b'al'), freq=13805
# [2026-03-25 23:42:03] Merge 20100/31743: best_pair=(b' Core', b'y'), freq=13689
# [2026-03-25 23:42:04] Merge 20200/31743: best_pair=(b' ele', b'ven'), freq=13575
# [2026-03-25 23:42:04] Merge 20300/31743: best_pair=(b' Hay', b'es'), freq=13448
# [2026-03-25 23:42:05] Merge 20400/31743: best_pair=(b'H', b'ard'), freq=13342
# [2026-03-25 23:42:05] Merge 20500/31743: best_pair=(b's', b'ch'), freq=13237
# [2026-03-25 23:42:06] Merge 20600/31743: best_pair=(b'it', b'em'), freq=13121
# [2026-03-25 23:42:06] Merge 20700/31743: best_pair=(b' B', b'orn'), freq=13028
# [2026-03-25 23:42:07] Merge 20800/31743: best_pair=(b' C', b'ord'), freq=12919
# [2026-03-25 23:42:07] Merge 20900/31743: best_pair=(b' retrie', b've'), freq=12804
# [2026-03-25 23:42:08] Merge 21000/31743: best_pair=(b' ho', b'ax'), freq=12694
# [2026-03-25 23:42:08] Merge 21100/31743: best_pair=(b' film', b'makers'), freq=12592
# [2026-03-25 23:42:09] Merge 21200/31743: best_pair=(b' s', b'ung'), freq=12507
# [2026-03-25 23:42:09] Merge 21300/31743: best_pair=(b'mark', b'ed'), freq=12410
# [2026-03-25 23:42:10] Merge 21400/31743: best_pair=(b' overt', b'urned'), freq=12329
# [2026-03-25 23:42:10] Merge 21500/31743: best_pair=(b'Cons', b'ider'), freq=12240
# [2026-03-25 23:42:10] Merge 21600/31743: best_pair=(b'qu', b'et'), freq=12152
# [2026-03-25 23:42:11] Merge 21700/31743: best_pair=(b' st', b'umbled'), freq=12053
# [2026-03-25 23:42:11] Merge 21800/31743: best_pair=(b' l', b'ith'), freq=11985
# [2026-03-25 23:42:12] Merge 21900/31743: best_pair=(b' B', b'ast'), freq=11913
# [2026-03-25 23:42:12] Merge 22000/31743: best_pair=(b' advert', b'ised'), freq=11831
# [2026-03-25 23:42:13] Merge 22100/31743: best_pair=(b'the', b'ir'), freq=11736
# [2026-03-25 23:42:13] Merge 22200/31743: best_pair=(b'l', b'ove'), freq=11660
# [2026-03-25 23:42:13] Merge 22300/31743: best_pair=(b' goal', b'ie'), freq=11567
# [2026-03-25 23:42:14] Merge 22400/31743: best_pair=(b' L', b'SU'), freq=11492
# [2026-03-25 23:42:14] Merge 22500/31743: best_pair=(b' success', b'ive'), freq=11414
# [2026-03-25 23:42:15] Merge 22600/31743: best_pair=(b'Des', b'ign'), freq=11330
# [2026-03-25 23:42:15] Merge 22700/31743: best_pair=(b'Reg', b'ardless'), freq=11236
# [2026-03-25 23:42:16] Merge 22800/31743: best_pair=(b' inflamm', b'ation'), freq=11154
# [2026-03-25 23:42:16] Merge 22900/31743: best_pair=(b' pair', b'ing'), freq=11064
# [2026-03-25 23:42:17] Merge 23000/31743: best_pair=(b' at', b'om'), freq=10976
# [2026-03-25 23:42:17] Merge 23100/31743: best_pair=(b' NAS', b'CAR'), freq=10894
# [2026-03-25 23:42:17] Merge 23200/31743: best_pair=(b' Resp', b'onse'), freq=10800
# [2026-03-25 23:42:18] Merge 23300/31743: best_pair=(b' Os', b'ama'), freq=10709
# [2026-03-25 23:42:18] Merge 23400/31743: best_pair=(b' discour', b'age'), freq=10628
# [2026-03-25 23:42:19] Merge 23500/31743: best_pair=(b'reat', b'h'), freq=10545
# [2026-03-25 23:42:19] Merge 23600/31743: best_pair=(b' bl', b'ur'), freq=10481
# [2026-03-25 23:42:19] Merge 23700/31743: best_pair=(b' k', b'idding'), freq=10414
# [2026-03-25 23:42:20] Merge 23800/31743: best_pair=(b' B', b'ath'), freq=10330
# [2026-03-25 23:42:20] Merge 23900/31743: best_pair=(b' Black', b's'), freq=10258
# [2026-03-25 23:42:21] Merge 24000/31743: best_pair=(b'ud', b'o'), freq=10189
# [2026-03-25 23:42:21] Merge 24100/31743: best_pair=(b'ilib', b'rium'), freq=10121
# [2026-03-25 23:42:21] Merge 24200/31743: best_pair=(b' DI', b'Y'), freq=10058
# [2026-03-25 23:42:22] Merge 24300/31743: best_pair=(b' class', b'ics'), freq=9990
# [2026-03-25 23:42:22] Merge 24400/31743: best_pair=(b'ha', b'us'), freq=9910
# [2026-03-25 23:42:23] Merge 24500/31743: best_pair=(b' Man', b'ila'), freq=9851
# [2026-03-25 23:42:23] Merge 24600/31743: best_pair=(b' Bar', b'cl'), freq=9775
# [2026-03-25 23:42:24] Merge 24700/31743: best_pair=(b' qu', b'oting'), freq=9712
# [2026-03-25 23:42:44] Merge 24800/31743: best_pair=(b' We', b'aver'), freq=9657
# [2026-03-25 23:42:45] Merge 24900/31743: best_pair=(b'A', b'I'), freq=9600
# [2026-03-25 23:42:45] Merge 25000/31743: best_pair=(b' A', b'uckland'), freq=9539
# [2026-03-25 23:42:46] Merge 25100/31743: best_pair=(b'Jon', b'athan'), freq=9482
# [2026-03-25 23:42:46] Merge 25200/31743: best_pair=(b' F', b'ut'), freq=9429
# [2026-03-25 23:42:46] Merge 25300/31743: best_pair=(b'id', b'ian'), freq=9363
# [2026-03-25 23:42:47] Merge 25400/31743: best_pair=(b'ut', b'on'), freq=9305
# [2026-03-25 23:42:47] Merge 25500/31743: best_pair=(b' Thom', b'son'), freq=9244
# [2026-03-25 23:42:47] Merge 25600/31743: best_pair=(b' list', b'ener'), freq=9192
# [2026-03-25 23:42:48] Merge 25700/31743: best_pair=(b'2', b'30'), freq=9134
# [2026-03-25 23:42:48] Merge 25800/31743: best_pair=(b'du', b'ino'), freq=9070
# [2026-03-25 23:42:48] Merge 25900/31743: best_pair=(b'\xe7\x9a', b'\x84'), freq=9005
# [2026-03-25 23:42:49] Merge 26000/31743: best_pair=(b' C', b'oca'), freq=8944
# [2026-03-25 23:42:50] Merge 26100/31743: best_pair=(b' cl', b'own'), freq=8889
# [2026-03-25 23:42:50] Merge 26200/31743: best_pair=(b'ting', b'u'), freq=8830
# [2026-03-25 23:42:50] Merge 26300/31743: best_pair=(b' St', b'orage'), freq=8779
# [2026-03-25 23:42:51] Merge 26400/31743: best_pair=(b' am', b'using'), freq=8740
# [2026-03-25 23:42:51] Merge 26500/31743: best_pair=(b' Al', b'onso'), freq=8682
# [2026-03-25 23:42:52] Merge 26600/31743: best_pair=(b' piss', b'ed'), freq=8639
# [2026-03-25 23:42:52] Merge 26700/31743: best_pair=(b' neat', b'ly'), freq=8575
# [2026-03-25 23:42:53] Merge 26800/31743: best_pair=(b' Ch', b'arge'), freq=8521
# [2026-03-25 23:42:53] Merge 26900/31743: best_pair=(b' worth', b'less'), freq=8476
# [2026-03-25 23:42:53] Merge 27000/31743: best_pair=(b' Car', b'negie'), freq=8424
# [2026-03-25 23:42:54] Merge 27100/31743: best_pair=(b' v', b'iability'), freq=8373
# [2026-03-25 23:42:54] Merge 27200/31743: best_pair=(b'greg', b'ated'), freq=8322
# [2026-03-25 23:42:54] Merge 27300/31743: best_pair=(b' work', b'flow'), freq=8276
# [2026-03-25 23:42:55] Merge 27400/31743: best_pair=(b' comm', b'uters'), freq=8228
# [2026-03-25 23:42:55] Merge 27500/31743: best_pair=(b'erc', b'ise'), freq=8171
# [2026-03-25 23:42:56] Merge 27600/31743: best_pair=(b' mus', b'cular'), freq=8123
# [2026-03-25 23:42:56] Merge 27700/31743: best_pair=(b' Gun', b's'), freq=8081
# [2026-03-25 23:42:56] Merge 27800/31743: best_pair=(b' Mal', b'ik'), freq=8031
# [2026-03-25 23:42:57] Merge 27900/31743: best_pair=(b' log', b'os'), freq=7986
# [2026-03-25 23:42:57] Merge 28000/31743: best_pair=(b' Sunder', b'land'), freq=7935
# [2026-03-25 23:42:58] Merge 28100/31743: best_pair=(b' hes', b'itant'), freq=7888
# [2026-03-25 23:42:58] Merge 28200/31743: best_pair=(b' Sil', b'ent'), freq=7844
# [2026-03-25 23:42:58] Merge 28300/31743: best_pair=(b' Roman', b'ian'), freq=7798
# [2026-03-25 23:42:59] Merge 28400/31743: best_pair=(b' Pars', b'ons'), freq=7750
# [2026-03-25 23:42:59] Merge 28500/31743: best_pair=(b'leg', b'ates'), freq=7703
# [2026-03-25 23:42:59] Merge 28600/31743: best_pair=(b' voy', b'age'), freq=7650
# [2026-03-25 23:43:00] Merge 28700/31743: best_pair=(b' t', b'id'), freq=7602
# [2026-03-25 23:43:00] Merge 28800/31743: best_pair=(b' anonym', b'ously'), freq=7558
# [2026-03-25 23:43:00] Merge 28900/31743: best_pair=(b'zz', b'a'), freq=7513
# [2026-03-25 23:43:01] Merge 29000/31743: best_pair=(b' impart', b'ial'), freq=7465
# [2026-03-25 23:43:01] Merge 29100/31743: best_pair=(b'ens', b'ing'), freq=7414
# [2026-03-25 23:43:02] Merge 29200/31743: best_pair=(b'27', b'0'), freq=7367
# [2026-03-25 23:43:02] Merge 29300/31743: best_pair=(b' f', b'oo'), freq=7320
# [2026-03-25 23:43:03] Merge 29400/31743: best_pair=(b'ge', b'bra'), freq=7280
# [2026-03-25 23:43:03] Merge 29500/31743: best_pair=(b' sign', b'ings'), freq=7241
# [2026-03-25 23:43:03] Merge 29600/31743: best_pair=(b' fare', b'well'), freq=7198
# [2026-03-25 23:43:04] Merge 29700/31743: best_pair=(b'arch', b's'), freq=7156
# [2026-03-25 23:43:04] Merge 29800/31743: best_pair=(b'ann', b'ie'), freq=7122
# [2026-03-25 23:43:04] Merge 29900/31743: best_pair=(b'ret', b'ch'), freq=7081
# [2026-03-25 23:43:05] Merge 30000/31743: best_pair=(b' hon', b'oring'), freq=7045
# [2026-03-25 23:43:05] Merge 30100/31743: best_pair=(b's', b'ac'), freq=7011
# [2026-03-25 23:43:06] Merge 30200/31743: best_pair=(b' dysfunction', b'al'), freq=6975
# [2026-03-25 23:43:06] Merge 30300/31743: best_pair=(b' Dev', b'ice'), freq=6935
# [2026-03-25 23:43:06] Merge 30400/31743: best_pair=(b' square', b'ly'), freq=6895
# [2026-03-25 23:43:07] Merge 30500/31743: best_pair=(b' eth', b'os'), freq=6856
# [2026-03-25 23:43:07] Merge 30600/31743: best_pair=(b' cr', b'umbling'), freq=6817
# [2026-03-25 23:43:08] Merge 30700/31743: best_pair=(b' F', b'TC'), freq=6777
# [2026-03-25 23:43:08] Merge 30800/31743: best_pair=(b'ordin', b'ates'), freq=6736
# [2026-03-25 23:43:09] Merge 30900/31743: best_pair=(b' cripp', b'ling'), freq=6700
# [2026-03-25 23:43:09] Merge 31000/31743: best_pair=(b'C', b'ook'), freq=6661
# [2026-03-25 23:43:09] Merge 31100/31743: best_pair=(b' mor', b'p'), freq=6629
# [2026-03-25 23:43:10] Merge 31200/31743: best_pair=(b'y', b'y'), freq=6592
# [2026-03-25 23:43:10] Merge 31300/31743: best_pair=(b' sl', b'ows'), freq=6553
# [2026-03-25 23:43:11] Merge 31400/31743: best_pair=(b'Fr', b'anc'), freq=6515
# [2026-03-25 23:43:11] Merge 31500/31743: best_pair=(b' compl', b'icit'), freq=6483
# [2026-03-25 23:43:11] Merge 31600/31743: best_pair=(b' dur', b'ability'), freq=6445
# [2026-03-25 23:43:12] Merge 31700/31743: best_pair=(b' bill', b'board'), freq=6412
# [2026-03-25 23:43:21] BPE merges completed.
# [2026-03-25 23:43:21] 词表已保存至: /Users/dami/cs336/cs336-assignment1-basics/data/owt_train.vocab.json
# [2026-03-25 23:43:21] 合并规则已保存至: /Users/dami/cs336/cs336-assignment1-basics/data/owt_train.merges.json
# [2026-03-25 23:43:21] 词表中最长的 token 长度: 64
# [2026-03-25 23:43:21] 内容: ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ
# [2026-03-25 23:43:21] run_train_bpe [OWT | Training] 执行完成。
# [2026-03-25 23:43:21] 耗时: 775.12 秒 (约 12.92 分钟)
# [2026-03-25 23:43:21] 峰值内存占用: 13974.02 MB
# -------------------------------------------------------------------------------------------------------------------------