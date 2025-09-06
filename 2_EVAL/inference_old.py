import torch
import time 
import pandas as pd
import gc  
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import os
import torch

from transformers.utils import logging
logging.set_verbosity_error()


# Ative otimizações compatíveis
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.matmul.allow_tf32 = True

try:
    from unsloth import FastLanguageModel  # Tem que estar no topo!
    USE_UNSLOTH = True
except ImportError:
    USE_UNSLOTH = False

def batch_infer(prompts, model, tokenizer, device, deterministic=True):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        if deterministic:
            outputs = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=5,
                num_return_sequences=1
            )
        else:
            outputs = model.generate(
                **inputs,
                do_sample=True,
                max_new_tokens=5,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                num_return_sequences=1
            )

    return [tokenizer.decode(o, skip_special_tokens=True).strip() for o in outputs]

model_names = [
    #"annagoncalves2/chatbot-gemma-3-12b-it-bnb-4bit-test",
    "annagoncalves2/chatbot-Llama-3.1-8B-unsloth-bnb-4bit-test",
    "annagoncalves2/chatbot-Llama-3.2-3B-Instruct-unsloth-bnb-4bit-test",
    "annagoncalves2/chatbot-phi-4-unsloth-bnb-4bit-test",
    "annagoncalves2/chatbot-Qwen2.5-7B-Instruct-bnb-4bit-test",
    "annagoncalves2/chatbot-Qwen2.5-14B-Instruct-unsloth-bnb-4bit-test",
    "annagoncalves2/chatbot-zephyr-sft-bnb-4bit-test",
    "annagoncalves2/chatbot-DeepSeek-R1-Distill-Llama-8B-test",
    "annagoncalves2/chatbot-Llama-3.1-8B-unsloth-bnb-4bit-V2",
    "annagoncalves2/chatbot-Llama-3.2-3B-Instruct-unsloth-bnb-4bit-V2",
    "annagoncalves2/chatbot-phi-4-unsloth-bnb-4bit-V2",
    "annagoncalves2/chatbot-Qwen2.5-7B-Instruct-bnb-4bit-V2",
    "annagoncalves2/chatbot-Qwen2.5-14B-Instruct-unsloth-bnb-4bit-V2",
    "annagoncalves2/chatbot-zephyr-sft-bnb-4bit-V2",
    "annagoncalves2/chatbot-DeepSeek-R1-Distill-Llama-8B-V2",
    #"annagoncalves2/chatbot-gemma-3-12b-it-bnb-4bit-V2",

]

run_names = [
    #"gemma-3-12b-it-bnb-4bit-V1",
    "Llama-3.1-8B-unsloth-bnb-4bit-V1",
    "Llama-3.2-3B-Instruct-unsloth-bnb-4bit-V1",
    "phi-4-unsloth-bnb-4bit-V1",
    "Qwen2.5-7B-Instruct-bnb-4bit-V1",
    "Qwen2.5-14B-Instruct-unsloth-bnb-4bit-V1",
    "zephyr-sft-bnb-4bit-V1",
    "DeepSeek-R1-Distill-Llama-8B-V1",
    "Llama-3.1-8B-unsloth-bnb-4bit-V2",
    "Llama-3.2-3B-Instruct-unsloth-bnb-4bit-V2",
    "phi-4-unsloth-bnb-4bit-V2",
    "Qwen2.5-7B-Instruct-bnb-4bit-V2",
    "Qwen2.5-14B-Instruct-unsloth-bnb-4bit-V2",
    "zephyr-sft-bnb-4bit-V2",
    "DeepSeek-R1-Distill-Llama-8B-V2",
    #"gemma-3-12b-it-bnb-4bit-V2",
]


token = ""
ds_org_eng = "/home/annap/Documents/chatbot_copy/DATASETS/MMLU/0_mmlu_prompt_pt.csv"
ds_output = "/home/annap/Documents/chatbot_copy/INFERENCE/DATASETS/UNSLOTH/"
base_model_dir = "./models"

dataset = pd.read_csv(ds_org_eng)
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 2
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

for model_path, run_name in tqdm(zip(model_names, run_names), total=len(model_names), desc="Processando Modelos"):
    print(f"🚀 Iniciando o job para o modelo: {run_name}")
    start_time = time.time()

    wandb.init(
        project="Unsloth Eval MMLU - EN-PT-PT",
        name=run_name,
        config={
            "model_name": model_path,
            "run_name": run_name,
            "batch_size": batch_size,
            "max_new_tokens": 5,
            "do_sample": False
        }
    )

    torch.cuda.empty_cache()
    gc.collect()

    local_model_dir = os.path.join(base_model_dir, model_path.split('/')[-1])
    use_local = os.path.exists(local_model_dir)
    model_source = local_model_dir if use_local else model_path

    try:
        if USE_UNSLOTH and "unsloth" in model_path.lower():
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_source,
                max_seq_length=4096,
                dtype=dtype,
                device_map="auto",
                token=token,
                trust_remote_code=True,
                local_files_only=use_local
            )
        else:
            config = AutoConfig.from_pretrained(model_source, token=token, trust_remote_code=True, local_files_only=use_local)
            model = AutoModelForCausalLM.from_pretrained(
                model_source,
                torch_dtype=dtype,
                use_auth_token=token,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=use_local
            )
            tokenizer = AutoTokenizer.from_pretrained(model_source, token=token, local_files_only=use_local)

        tokenizer.padding_side = "left"

    except Exception as e:
        print(f"❌ Falha ao carregar o modelo {model_path}: {e}")
        wandb.finish()
        continue

    dataloader = DataLoader(dataset['prompt'].tolist(), batch_size=batch_size, shuffle=False)
    results = []

    for batch in tqdm(dataloader, desc=f"Inferindo com {run_name}"):
        results.extend(batch_infer(batch, model, tokenizer, device, deterministic=True))
        torch.cuda.empty_cache()
        gc.collect()

    dataset['output_model'] = results
    dataset.to_csv(os.path.join(ds_output, f"{run_name}.csv"), index=False)

    elapsed_time = time.time() - start_time
    print(f"✅ Tempo de execução: {elapsed_time:.2f}s")
    wandb.log({"elapsed_time_sec": elapsed_time})
    wandb.finish()

    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

print("🎉 Todos os modelos foram inferidos com sucesso.")
