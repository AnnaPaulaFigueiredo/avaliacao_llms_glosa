import os
import warnings
import gc
import time

import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.utils import logging
import wandb

# Suppress warnings and unnecessary logs
warnings.filterwarnings("ignore")
logging.set_verbosity_error()
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Enable recommended CUDA optimizations
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.matmul.allow_tf32 = True

# ✅ Verifica se o modelo requer FP8 e se a GPU suporta
def is_model_compatible(config):
    if hasattr(config, "quantization_config"):
        quant_config = config.quantization_config
        if "fp8" in str(quant_config).lower():
            cc = torch.cuda.get_device_capability()
            if cc[0] + cc[1] / 10 < 8.9:  # Exige compute capability >= 8.9 (ex: 4090)
                return False
    return True

# der erro lembre-se de trocar o batch infer

def batch_infer(prompts, model, tokenizer, device):
    try:
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

        with torch.no_grad():
            
            outputs = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=5,
            num_return_sequences=1
        )


        decoded = [tokenizer.decode(output, skip_special_tokens=True).strip().lower() for output in outputs]

        return decoded

    except torch.cuda.OutOfMemoryError:
        print("🚨 ERRO: CUDA OUT OF MEMORY! Reduzindo batch_size e limpando cache...")
        torch.cuda.empty_cache()
        gc.collect()
        return ["Erro OOM"] * len(prompts)


# ajustado para o gemma
'''
def batch_infer(prompts, model, tokenizer, _):  # o device externo não é mais necessário
    try:
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)

        # Detecta automaticamente o device correto do modelo
        model_device = next(model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=5,
                num_return_sequences=1
            )

        decoded = [tokenizer.decode(output, skip_special_tokens=True).strip().lower() for output in outputs]
        return decoded

    except torch.cuda.OutOfMemoryError:
        print("🚨 ERRO: CUDA OUT OF MEMORY! Reduzindo batch_size e limpando cache...")
        torch.cuda.empty_cache()
        gc.collect()
        return ["Erro OOM"] * len(prompts)

'''
# Modelos para inferência # 840h estimadas para inferencia :/
'''
model_names = [
    #"unsloth/zephyr-sft-bnb-4bit",                             # ~sem tamanho explícito (provavelmente pequeno)
    #"unsloth/phi-4-unsloth-bnb-4bit",                         # ~4B
    #"unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit",         # ~3B
    #"unsloth/Qwen2.5-7B-Instruct-bnb-4bit",                   # ~7B
    #"unsloth/Llama-3.1-8B-unsloth-bnb-4bit",                  # ~8B
    #"unsloth/DeepSeek-R1-Distill-Llama-8B",                   # ~8B (distilled)
    #"unsloth/gemma-3-12b-it-bnb-4bit",                        # ~12B
    #"unsloth/Qwen2.5-14B-Instruct-unsloth-bnb-4bit",          # ~14B
    #"unsloth/DeepSeek-V3-bf16",                               # ~grande (não especificado, assume grande)
    #"unsloth/Llama-3.2-11B-Vision-unsloth-bnb-4bit",          # ~11B
    "unsloth/Qwen3-14B-unsloth-bnb-4bit",                     # ~14B
    "unsloth/Qwen2.5-32B-Instruct-bnb-4bit",                  # ~32B
    #"unsloth/gemma-3-27b-it-unsloth-bnb-4bit",                # ~27B
    #"unsloth/Llama-4-Scout-17B-16E-Instruct-unsloth-bnb-4bit",# ~17B
    #"unsloth/Llama-3.3-70B-Instruct-bnb-4bit",                # ~70B
    #"unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit",           # ~70B
    #"unsloth/Qwen3-32B-unsloth-bnb-4bit"                      # ~32B
]

 
run_names = [
    #"zephyr-sft-bnb-4bit",                             # ~sem tamanho explícito (provavelmente pequeno)
    #"phi-4-unsloth-bnb-4bit",                         # ~4B
    #"Llama-3.2-3B-Instruct-unsloth-bnb-4bit",         # ~3B
    #"Qwen2.5-7B-Instruct-bnb-4bit",                   # ~7B
    #"Llama-3.1-8B-unsloth-bnb-4bit",                  # ~8B
    #"DeepSeek-R1-Distill-Llama-8B",                   # ~8B (distilled)
    #"gemma-3-12b-it-unsloth-bnb-4bit",                # ~12B
    #"Qwen2.5-14B-Instruct-unsloth-bnb-4bit",          # ~14B
    #"DeepSeek-V3-bf16-unsloth-largest",               # ~grande (não especificado, assume grande)
    #"Llama-3.2-11B-Vision-unsloth-bnb-4bit-unsloth-largest",  # ~11B
    "Qwen3-14B-unsloth-bnb-4bit-unsloth-largest",     # ~14B
    "Qwen2.5-32B-Instruct-bnb-4bit-unsloth-largest",  # ~32B
]
'''

model_names = [
    #"annagoncalves2/chatbot-gemma-3-12b-it-bnb-4bit-test",
    #"annagoncalves2/chatbot-Llama-3.1-8B-unsloth-bnb-4bit-test",
    #"annagoncalves2/chatbot-Llama-3.2-3B-Instruct-unsloth-bnb-4bit-test",
    #"annagoncalves2/chatbot-phi-4-unsloth-bnb-4bit-test",
    #"annagoncalves2/chatbot-Qwen2.5-7B-Instruct-bnb-4bit-test",
    #"annagoncalves2/chatbot-Qwen2.5-14B-Instruct-unsloth-bnb-4bit-test",
    #"annagoncalves2/chatbot-zephyr-sft-bnb-4bit-test",
    #"annagoncalves2/chatbot-DeepSeek-R1-Distill-Llama-8B-test", #passar de novo para 
    #"annagoncalves2/chatbot-Llama-3.1-8B-unsloth-bnb-4bit-V2",
    #"annagoncalves2/chatbot-Llama-3.2-3B-Instruct-unsloth-bnb-4bit-V2",
    #"annagoncalves2/chatbot-phi-4-unsloth-bnb-4bit-V2",
    #"annagoncalves2/chatbot-Qwen2.5-7B-Instruct-bnb-4bit-V2",
    #"annagoncalves2/chatbot-Qwen2.5-14B-Instruct-unsloth-bnb-4bit-V2",
    #"annagoncalves2/chatbot-zephyr-sft-bnb-4bit-V2",
    #"annagoncalves2/chatbot-DeepSeek-R1-Distill-Llama-8B-V2", # passar de novo
    "annagoncalves2/chatbot-gemma-3-12b-it-bnb-4bit-V2",
]

run_names = [
    #"gemma-3-12b-it-bnb-4bit-V1",
    #"Llama-3.1-8B-unsloth-bnb-4bit-V1",
    #"Llama-3.2-3B-Instruct-unsloth-bnb-4bit-V1",
    #"phi-4-unsloth-bnb-4bit-V1",
    #"Qwen2.5-7B-Instruct-bnb-4bit-V1",
    #"Qwen2.5-14B-Instruct-unsloth-bnb-4bit-V1",
    #"zephyr-sft-bnb-4bit-V1",
    #"DeepSeek-R1-Distill-Llama-8B-V1",
    #"Llama-3.1-8B-unsloth-bnb-4bit-V2",
    #"Llama-3.2-3B-Instruct-unsloth-bnb-4bit-V2",
    #"phi-4-unsloth-bnb-4bit-V2",
    #"Qwen2.5-7B-Instruct-bnb-4bit-V2",
    #"Qwen2.5-14B-Instruct-unsloth-bnb-4bit-V2",
    #"zephyr-sft-bnb-4bit-V2",
    #"DeepSeek-R1-Distill-Llama-8B-V2",
    "gemma-3-12b-it-bnb-4bit-V2"
]

token = ""

ds_org_eng = "/home/annap/Documents/chatbot_copy/DATASETS/MMLU/0_mmlu_prompt_english.csv"

#   0_mmlu_prompt_glosa_pt.csv 
#   0_mmlu_prompt_pt.csv ok
#   0_mmlu_prompt_glosa.csv ok 
#   0_mmlu_prompt_english.csv ok
ds_output = "/home/annap/Documents/chatbot_copy/INFERENCE/DATASETS/UNSLOTH/V2/ENG/"

dataset = pd.read_csv(ds_org_eng)

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 2

torch.backends.cuda.matmul.allow_tf32 = True

# Adicionando tqdm para monitorar o loop dos modelos
for model_path, run_name in tqdm(zip(model_names, run_names), total=len(model_names), desc="Processando Modelos"):
    print(f"\n🚀 Iniciando o job para o modelo: {run_name}")
    start_time = time.time()

    try:
        wandb.init(
            project="Unsloth Eval MMLU - PORTUGUESE", 
            name=f"{run_name}",
            config={
                "model_name": model_path,
                "run_name": run_name,
                "batch_size": batch_size,
                "max_new_tokens": 5,
                "temperature": 0.0,
                "top_p": 1.0
            }
        )

        torch.cuda.empty_cache()
        gc.collect()

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        config = AutoConfig.from_pretrained(model_path, token=token, trust_remote_code=True)

        # ❌ Skip se não for compatível com a GPU (ex: FP8)
        if not is_model_compatible(config):
            print(f"❌ Modelo {run_name} exige hardware não suportado (ex: FP8). Pulando.")
            wandb.finish()
            continue

        '''  model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            use_auth_token=True,  # Ou token=token
            device_map="auto",
        )'''
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
            token=token,
            config={"model_type": "llama"}  
        )

        tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
        
        tokenizer.padding_side = "left"
        dataloader = DataLoader(dataset['prompt'].tolist(), batch_size=batch_size, shuffle=False)

        results = []
        for batch in tqdm(dataloader, desc=f"Inferindo com {run_name}"):
            results.extend(batch_infer(batch, model, tokenizer, device))
            torch.cuda.empty_cache()
            gc.collect()

        # Cria um DataFrame apenas com as colunas desejadas
        output_df = pd.DataFrame({
            'ID': dataset['ID'],
            'prompt': dataset['prompt'],
            'output_model': results
        })

        output_df.to_csv(ds_output + f"{run_name}.csv", index=False)

        elapsed_time = time.time() - start_time
        formatted_time = time.strftime('%H:%M:%S', time.gmtime(elapsed_time))
        print(f"⏳ Tempo de execução para {run_name}: {formatted_time}")

        # Loga no Weights & Biases
        wandb.log({
            "elapsed_time_sec": elapsed_time,
            "elapsed_time_hms": formatted_time
        })

        wandb.finish()

        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()

    except torch.cuda.OutOfMemoryError:
        print(f"❌ OOM: Modelo {run_name} excedeu a memória da GPU.")
        wandb.finish()
        torch.cuda.empty_cache()
        gc.collect()
        continue

    except Exception as e:
        print(f"❌ Erro ao processar o modelo {run_name}: {e}")
        wandb.finish()
        torch.cuda.empty_cache()
        gc.collect()
        continue

print("🎉 Todos os modelos Unsloth processados!")
