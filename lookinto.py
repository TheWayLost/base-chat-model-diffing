import os
from sae_lens import SAE, SAEConfig
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from datasets import load_dataset, load_from_disk, concatenate_datasets, IterableDataset
from transformer_lens import HookedTransformer, utils

# --- (保留原有的 load_sae, get_feature_vectors, calculate_cosine_similarity 函数) ---
def load_sae(path: str, device="cpu") -> SAE:
    if not os.path.exists(path):
        raise ValueError(f"路径不存在: {path}")
    cfg_path = os.path.join(path, "cfg.json")
    if not os.path.exists(cfg_path):
        raise ValueError(f"配置文件不存在: {cfg_path}")
    sae = SAE.load_from_pretrained(
        path = path,
        device = device
    )
    return sae

def get_feature_vectors(sae):
    W_dec = sae.W_dec.detach().cpu()
    W_dec_norm = F.normalize(W_dec, p=2, dim=1)
    return W_dec_norm

def calculate_cosine_similarity(vectors1, vectors2):
    cosine_similarities = torch.einsum("id,id->i", vectors1, vectors2)
    return cosine_similarities

def calculate_feature_rotation(sae_start, sae_final):
    feature_vectors_start = get_feature_vectors(sae_start)
    feature_vectors_final = get_feature_vectors(sae_final)
    cosine_sim = calculate_cosine_similarity(feature_vectors_start, feature_vectors_final)
    return cosine_sim

# 假设你的四个 SAE 模型分别位于以下路径：
path_sd = "/root/autodl-tmp/huggingface/qwen1.5-0.5b-alpaca/"
path_sm = "/root/autodl-tmp/huggingface/retrial/"
path_df = "/root/autodl-tmp/huggingface/qwen-basechatsae-post/"
path_mf = "/root/autodl-tmp/huggingface/retrial-from-forumsae-post/"

# 加载四个 SAE 模型
device = "cuda" if torch.cuda.is_available() else "cpu"
sae_sd = load_sae(path_sd, device=device)
sae_sm = load_sae(path_sm, device=device)
sae_df = load_sae(path_df, device=device)
sae_mf = load_sae(path_mf, device=device)
print("SAE 模型加载完成！")

# 1. Data-first path (S→D→F)
cosine_sim_sdf = calculate_feature_rotation(sae_sd, sae_df)
# 2. Model-first path (S→M→F)
cosine_sim_smf = calculate_feature_rotation(sae_sm, sae_mf)

# 绘制散点图
plt.figure(figsize=(8, 8))
plt.scatter(cosine_sim_sdf, cosine_sim_smf, s=5, alpha=0.7)
plt.xlabel("cos(S→D, D→F)")
plt.ylabel("cos(S→M, M→M→F)")
plt.title("Stage-Wise Feature Rotation")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.axhline(y=0.7, color='r', linestyle='--', linewidth=0.5)
plt.axvline(x=0.7, color='r', linestyle='--', linewidth=0.5)
plt.savefig("my_plot2.png")

# 找出符合条件的特征
threshold = 0.7
interesting_features = (cosine_sim_sdf < threshold) & (cosine_sim_smf < threshold)
interesting_feature_indices = torch.nonzero(interesting_features).squeeze()
print(f"Interesting feature indices: {interesting_feature_indices}")
print("图表已保存")

##################### 新增部分 (预先计算并存储激活值) ##################################

from datasets import load_dataset, load_from_disk
from transformer_lens import HookedTransformer, utils
import numpy as np

# 加载模型和分词器 (使用 Qwen/Qwen1.5-0.5B-Chat)
model_name = "Qwen/Qwen1.5-0.5B-Chat"
model = HookedTransformer.from_pretrained(model_name, device=device, trust_remote_code=True)
tokenizer = model.tokenizer

# 找到余弦相似度最小的 N 个特征
# n_top_features = 10
# feature_change_metric = cosine_sim_sdf + cosine_sim_smf
# top_feature_indices = torch.argsort(feature_change_metric)[:n_top_features]

save_path = "/root/autodl-tmp/huggingface/my-activation/"

# 改为手动指定特征
target_features = [32232, 13945]
n_top_samples = 10  # 每个特征显示前 10 个样本
results_dir = "/root/workspace/diffing/result/"
os.makedirs(results_dir, exist_ok=True)

# 遍历这些特征，找到激活值最大的输入样本 (从保存的文件中加载激活值)
k=10
for feature_idx in target_features:
    
    print(f"\n--- Feature Index: {feature_idx} ---")

    top_k_activations = []  # (activation_value, sample_idx, pos_idx)

    for chunk_idx in range(4):
        # 加载当前块的 SAE 激活值
        chunk_activations = np.load(os.path.join(save_path, f"all_sae_activations_chunk{chunk_idx}.npy"))

        # 获取该特征在当前块中的激活值
        feature_activations_chunk = chunk_activations[:, :, feature_idx]  # (chunk_samples, pos)

        # 找到当前块中的最大激活值和索引
        chunk_max_activations = feature_activations_chunk.flatten()
        top_k_indices_in_chunk = np.argpartition(chunk_max_activations, -k)[-k:]  # 找到最大的k个值的索引
        
        for flat_idx in top_k_indices_in_chunk:
            sample_idx_in_chunk, pos_idx = np.unravel_index(flat_idx, feature_activations_chunk.shape)
            activation_value = feature_activations_chunk[sample_idx_in_chunk, pos_idx]
            
            sample_idx = chunk_idx * 500 + sample_idx_in_chunk
            top_k_activations.append((activation_value, sample_idx, pos_idx))

        del chunk_activations, feature_activations_chunk # 释放内存
    
    # Sort all activations (from all chunks)
    top_k_activations.sort(key=lambda x: x[0], reverse=True)  # 降序排列
    top_k_activations = top_k_activations[:k]  # Get top k
    

    # 获取对应的输入 token (需要重新加载数据集)
    if 'dataset' not in locals():
         dataset = load_dataset("Lanzo-T-H/tokenized_alpaca_qwen1.5-parquet", split="train")
         dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])


    # 打印上下文
    context_window = 10
    print("---------------------------")
    for activation_value, sample_idx, pos_idx in top_k_activations:
        input_tokens = dataset[int(sample_idx)]["input_ids"]

        start_idx = max(0, pos_idx - context_window)
        end_idx = min(len(input_tokens), pos_idx + context_window + 1)
        context_tokens = input_tokens[start_idx:end_idx]

        decoded_text = tokenizer.decode(context_tokens)
        decode_max_idx = tokenizer.decode(input_tokens[pos_idx:pos_idx + 1])

        print(f"Max Activation Value: \n{(start_idx, pos_idx, end_idx)}\n{decode_max_idx}\n{activation_value}")
        print(f"Context Tokens: \n{context_tokens}")
        print(f"Decoded Text:\n{decoded_text}")
        print("---------------------------")