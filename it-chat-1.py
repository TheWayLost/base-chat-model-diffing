import torch
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from sae_lens import LanguageModelSAERunnerConfig, SAETrainingRunner
from transformer_lens import HookedTransformer
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print("Using device:", device)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

total_training_steps = 100000
batch_size = 256
total_training_tokens = total_training_steps * batch_size

lr_warm_up_steps = 0
lr_decay_steps = total_training_steps // 5  # 20% of training
l1_warm_up_steps = total_training_steps // 20  # 5% of training

cfg = LanguageModelSAERunnerConfig(
    # Data Generating Function (Model + Training Distibuion)
    model_class_name="HookedTransformer",
    model_name="qwen1.5-0.5b-chat",  # our model (more options here: https://neelnanda-io.github.io/TransformerLens/generated/model_properties_table.html)
    hook_name="blocks.13.hook_resid_pre",  # A valid hook point (see more details here: https://neelnanda-io.github.io/TransformerLens/generated/demos/Main_Demo.html#Hook-Points)
    hook_layer=13,  # Only one layer in the model.
    hook_eval="NOT_IN_USE",
    d_in=1024,
    d_sae = 32768,
    activation_fn="relu",
    dataset_path="Lanzo-T-H/tokenized_alpaca_qwen1.5-parquet",
    is_dataset_tokenized=True,
    dataset_trust_remote_code = True,
    use_cached_activations=False,
    # SAE Parameters
    architecture="standard",
    mse_loss_normalization=None,  # We won't normalize the mse loss,  
    from_pretrained_path="/root/autodl-tmp/huggingface/qwen1.5-0.5b-alpaca/",
    normalize_sae_decoder=False,
    noise_scale= 0.0,
    decoder_orthogonal_init=False,
    decoder_heuristic_init=True,
    init_encoder_as_decoder_transpose=False,
    # Training Parameters
    lr=5e-5,
    adam_beta1=0.9,  # adam params (default, but once upon a time we experimented with these.)
    adam_beta2=0.999,
    lr_scheduler_name="constant",  # constant learning rate with warmup.
    lr_warm_up_steps=lr_warm_up_steps,  # this can help avoid too many dead features initially.
    lr_decay_steps=lr_decay_steps,  # this will help us avoid overfitting.
    l1_coefficient=5,  # will control how sparse the feature activations are
    l1_warm_up_steps=l1_warm_up_steps,  # this can help avoid too many dead features initially.
    lp_norm=1.0,  # the L1 penalty (and not a Lp for p < 1)
    train_batch_size_tokens=batch_size,
    context_size=256,  # will control the lenght of the prompts we feed to the model. Larger is better but slower. so for the tutorial we'll use a short one.
    # Activation Store Parameters
    n_batches_in_buffer=64,  # controls how many activations we store / shuffle.
    training_tokens=0,  # 100 million tokens is quite a few, but we want to see good stats. Get a coffee, come back.
    finetuning_tokens=total_training_tokens,
    store_batch_size_prompts=16,
    # Resampling protocol
    use_ghost_grads=False,  # we don't use ghost grads anymore.
    feature_sampling_window=1000,  # this controls our reporting of feature sparsity stats
    dead_feature_window=1000,  # would effect resampling or ghost grads if we were using it.
    dead_feature_threshold=1e-4,  # would effect resampling or ghost grads if we were using it.
    # WANDB
    log_to_wandb=True,  # always use wandb unless you are just testing code.
    wandb_project="sae_lens_tutorial",
    wandb_log_frequency=30,
    eval_every_n_wandb_logs=20,
    run_name="chatmodel + chatdata + basechatsae",
    # Misc
    device=device,
    seed=42,
    n_checkpoints=0,
    checkpoint_path="checkpoints",
    dtype="float32"
)
# model_path = "/root/autodl-tmp/huggingface/gemma-2-2b"
# tokenizer_2b = AutoTokenizer.from_pretrained(model_path)
# config_2b = AutoConfig.from_pretrained(model_path)
# hf_model_2b = AutoModelForCausalLM.from_pretrained(
#         model_path,
#         config=config_2b,
#     )
# cfg.model_from_pretrained_kwargs = {
#     "hf_model": hf_model_2b,
#     "tokenizer": tokenizer_2b,
# }

sparse_autoencoder = SAETrainingRunner(cfg).run()