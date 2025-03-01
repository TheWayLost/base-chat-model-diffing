from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import torch

# 1. 加载数据集 (这里使用 "tatsu-lab/alpaca" 作为示例，你可以替换为你自己的数据集)
try:
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
except Exception as e:
    print(f"加载数据集失败: {e}")
    print("请检查数据集名称和网络连接。")
    exit()

# 2. 定义聊天模板 (根据 Qwen1.5 的格式)
# 参考: https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat
# 以及 https://github.com/QwenLM/Qwen/blob/main/examples/tokenizer_showcase.py
# Qwen1.5 使用的特殊 token
# <|im_start|>  表示对话开始或系统消息
# <|im_end|>    表示对话结束或用户/助手消息结束
# <|endoftext|> 表示文本结束 (EOS)

def apply_chat_template(example):
    """
    将 Alpaca 数据集中的单个样本转换为 Qwen1.5 的聊天模板格式。

    Args:
        example: 数据集中的一个样本 (字典，包含 "instruction", "input", "output" 键)。

    Returns:
        格式化后的文本字符串。
    """
    # 系统消息 (通常是 instruction)
    system_message = f"<|im_start|>system\n{example['instruction']}<|im_end|>\n"

    # 用户消息 (input 不为空时)
    user_message = ""
    if example["input"]:  # Alpaca 数据集中，input 字段可能为空
        user_message = f"<|im_start|>user\n{example['input']}<|im_end|>\n"

    # 助手消息 (output)
    assistant_message = f"<|im_start|>assistant\n{example['output']}<|im_end|>\n"

    # 拼接成完整的对话
    formatted_text = system_message + user_message + assistant_message
    return {"text": formatted_text}


# 3. 应用聊天模板
dataset = dataset.map(apply_chat_template)

# 4. 分词
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B", trust_remote_code=True)
# Qwen1.5 tokenizer 需要设置 trust_remote_code=True

# 如果 tokenizer 没有自动添加 special tokens, 需要手动添加
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # 或者 "<|endoftext|>" , 取决于具体情况

def tokenize_function(examples):

    return tokenizer(examples["text"],
                     truncation=True,
                     max_length=512,  # 根据需要调整最大长度
                     padding="max_length", # 或者 "longest" 根据你的需求
                     return_tensors="pt")



tokenized_dataset = dataset.map(tokenize_function, batched=True)


# 5. (可选) 保存处理后的数据集
tokenized_dataset.save_to_disk("tokenized_alpaca_qwen1.5")

# # (可选) 加载处理后的数据集
# loaded_dataset = load_from_disk("tokenized_alpaca_qwen1.5")

# 验证
print(tokenized_dataset[0])

# 查看分词结果
print(tokenizer.decode(tokenized_dataset[0]['input_ids']))