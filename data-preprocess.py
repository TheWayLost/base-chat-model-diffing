import json
from sae_lens import PretokenizeRunner, PretokenizeRunnerConfig
from transformers import AutoTokenizer

# def convert_alpaca_to_gemma_for_pretokenizing(alpaca_example):
#     instruction = alpaca_example["instruction"]
#     input_text = alpaca_example.get("input", "")
#     output_text = alpaca_example["output"]
# 
#     user_turn = "<start_of_turn>user\n"
#     user_turn += instruction
#     if input_text:
#         user_turn += "\n" + input_text
#     user_turn += "<end_of_turn>\n"
# 
#     model_turn = "<start_of_turn>model\n"
#     model_turn += output_text
#     model_turn += "<end_of_turn>\n"
# 
#     return user_turn + model_turn
# 
# def preprocess_alpaca_data(input_file, output_file):
#     """
#     读取 Alpaca 格式的 JSON 文件，转换为 Gemma 格式，并保存为 JSONL 文件。
# 
#     Args:
#         input_file:  原始 Alpaca JSON 文件的路径。
#         output_file:  输出的 Gemma 格式 JSONL 文件的路径。
#     """
#     print("start")
#     with open(input_file, 'r', encoding='utf-8') as infile, \
#             open(output_file, 'w', encoding='utf-8') as outfile:
#         data = json.load(infile)
#         for example in data:
#             gemma_text = convert_alpaca_to_gemma_for_pretokenizing(example)
#             outfile.write(json.dumps({"text": gemma_text}) + '\n')
#     print("end")
# 
# 
# # 预处理数据集：
# input_alpaca_file = "/root/autodl-tmp/huggingface/alpaca-llama-factory/alpaca_en_demo.json"  # 原始 Alpaca 数据
# output_gemma_file = "/root/autodl-tmp/huggingface/alpaca-llama-factory/alpaca_preprocessed.jsonl"  # 预处理后的 Gemma 数据
# 
# preprocess_alpaca_data(input_alpaca_file, output_gemma_file)

# 预分词
# cfg = PretokenizeRunnerConfig()
# cfg.tokenizer_name = "/root/autodl-tmp/huggingface/gemma-2-2b" 
# cfg.dataset_path = "json"
# cfg.data_files = ["/root/autodl-tmp/huggingface/alpaca-llama-factory/alpaca_preprocessed.jsonl"] # 预处理后的数据
# cfg.column_name = "text"
# cfg.split = "train"
# cfg.context_size = 128  # 根据需要调整
# cfg.shuffle = True
# 
# #设置特殊token
# tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
# 
# cfg.begin_batch_token = tokenizer.bos_token_id
# cfg.begin_sequence_token = None # 对于gemma2来说不需要这个?
# cfg.sequence_separator_token = tokenizer.eos_token_id
# 
# # 选择保存方式 (二选一):
# # 方式一：保存到本地
# # cfg.save_path = "/root/autodl-tmp/pretokenized_alpaca"
# 
# # 方式二：上传到 Hugging Face Hub (需要先登录 `huggingface-cli login`)
# cfg.hf_repo_id = "Lanzo-T-H/alpaca_for_model_diff"
# cfg.hf_is_private_repo = False 
# 
# # 3. 创建 PretokenizeRunner 实例并运行
# tokenized_dataset = PretokenizeRunner(cfg).run()
# 
# print("预分词完成!")

import os
from datasets import load_dataset, load_from_disk

def convert_and_shard_to_parquet(dataset_path, output_dir, num_shards=1):
    """
    将数据集加载、转换为 Parquet 格式并分片保存。

    Args:
        dataset_path: 数据集路径 (可以是目录或文件)。
        output_dir:  输出目录。
        num_shards:  分片数量。
    """

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 尝试作为目录加载 (可能包含多个文件)
        dataset = load_from_disk(dataset_path)
    except FileNotFoundError:
        try:
            # 尝试作为单个文件加载
            dataset = load_dataset("json", data_files=dataset_path)  # 如果是单个 JSON 文件
        except Exception as e:
            print(f"无法加载数据集: {e}")
            return


    # 如果是 DatasetDict, 取第一个split (通常是 'train')
    if isinstance(dataset, dict): #check if dataset is a dictionary
        if len(dataset) > 0:
            first_key = next(iter(dataset)) #get first key in dictionary
            dataset = dataset[first_key] #get first element in the dataset
        else:
            print("Dataset dictionary is empty.")
            return

    # 分片并保存为 Parquet
    for shard_idx in range(num_shards):
        shard = dataset.shard(index=shard_idx, num_shards=num_shards)
        shard.to_parquet(os.path.join(output_dir, f"{shard_idx:05d}.parquet"))

    print(f"数据集已转换为 Parquet 格式并分片保存在: {output_dir}")

# 使用示例
dataset_path = "/root/autodl-tmp/huggingface/tokenized_alpaca_qwen1.5"  # 你的数据集路径
output_parquet_dir = "/root/autodl-tmp/huggingface/tokenized_alpaca_qwen1.5-parquet"  # 输出 Parquet 文件的目录
convert_and_shard_to_parquet(dataset_path, output_parquet_dir)

