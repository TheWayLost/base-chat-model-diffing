import torch
from transformer_lens import HookedTransformer
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch.nn as nn
import json


# class JumpReLUSAE(nn.Module):
#     def __init__(self, d_model, d_sae):
#         # Note that we initialise these to zeros because we're loading in pre-trained weights.
#         # If you want to train your own SAEs then we recommend using blah
#         super().__init__()
#         self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
#         self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))
#         self.threshold = nn.Parameter(torch.zeros(d_sae))
#         self.b_enc = nn.Parameter(torch.zeros(d_sae))
#         self.b_dec = nn.Parameter(torch.zeros(d_model))
# 
#     def encode(self, input_acts):
#         pre_acts = input_acts @ self.W_enc + self.b_enc
#         mask = (pre_acts > self.threshold)
#         acts = mask * torch.nn.functional.relu(pre_acts)
#         return acts
# 
#     def decode(self, acts):
#         return acts @ self.W_dec + self.b_dec
# 
#     def forward(self, acts):
#         acts = self.encode(acts)
#         recon = self.decode(acts)
#         return recon
# 
# 
# 定义设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# 加载 Gemma 2-2b 模型
model_path = "/root/autodl-tmp/huggingface/gemma-2-2b"
tokenizer_2b = AutoTokenizer.from_pretrained(model_path)
config_2b = AutoConfig.from_pretrained(model_path)
hf_model_2b = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config_2b,
    )
print("hf success1")
model_2b = HookedTransformer.from_pretrained(
        "google/gemma-2-2b",
        hf_model=hf_model_2b,
        device=device,
        tokenizer=tokenizer_2b,
        dtype=torch.float16,
    )
print("gemma-2-2b loaded successfully")


# 加载 Gemma 2-2b-it 模型
model_path_it = "/root/autodl-tmp/huggingface/gemma-2-2b-it"
config_2b_it = AutoConfig.from_pretrained(model_path_it)
tokenizer_2b_it = AutoTokenizer.from_pretrained(model_path_it)
hf_model_2b_it = AutoModelForCausalLM.from_pretrained(
        model_path_it,
        config=config_2b_it,
    )
print("hf success2")
model_2b_it = HookedTransformer.from_pretrained(
        model_name="google/gemma-2-2b-it",
        hf_model=hf_model_2b_it,
        device=device,
        tokenizer=tokenizer_2b_it,
        dtype=torch.float16,
    )
print("gemma-2-2b-it loaded successfully")

# 加载 SAE 的函数
# def load_sae():
#     # 加载 SAE 参数
#     params = np.load("/root/autodl-tmp/huggingface/gemma-scope-2b-pt-res/params.npz")
#     pt_params = {k: torch.from_numpy(v).to(device) for k, v in params.items()}
# 
#     sae = JumpReLUSAE(params['W_enc'].shape[0], params['W_enc'].shape[1])
#     sae.load_state_dict(pt_params)
#     sae.to(device)
#     sae.eval()  # 设置为评估模式
#     return sae
# 
# sae_2b = load_sae()
#from sae_lens import SAE
#
#
#sae, cfg_dict, sparsity = SAE.from_pretrained(
#    release="gemma-scope-2b-pt-res",  # see other options in sae_lens/pretrained_saes.yaml
#    sae_id="layer_12/width_16k/average_l0_82",  # won't always be a hook point
#    device="cuda",
#)
#
#print(cfg_dict)
#
#with open("/root/autodl-tmp/huggingface/gemma-scope-2b-pt-res/cfg.json", "w", encoding="utf-8") as f:
#    json.dump(cfg_dict, f, indent=4, ensure_ascii=False)
#
#print("cfg_dict 已保存到 cfg.json")
#
#
#print("gemma-scope-2b-pt-res/layer_12/width_16k/average_l0_82/params.npz loaded successfully")
#
#
#print("Models and SAE loaded successfully!")
#
