import pandas as pd
import requests
from tqdm import tqdm
import time
import gc
import wandb
from concurrent.futures import ThreadPoolExecutor, as_completed

# Nome do arquivo de entrada
INPUT_CSV = "/home/annap/Documents/chatbot_copy/DATASETS/MMLU/08_mmlu_formated_eval.csv"

models = [
    "deepseek-r1:7b",
    "deepseek-r1:32b",
    "gemma3:12b",
    "llama3.1:8b",
    "phi3:14b",
    "qwen2.5:14b",
    "qwen2.5:32b",
    "zephyr:7b"
]

run_names = [
    "deep_7b",
    "deep_32b",
    "gemma_12b",
    "llama_8b",
    "phi14b",
    "qwen_14b",
    "qwen_32b",
    "zephyr_7b"
]

def run_ollama(model, prompt):
    payload = {
        "model": model,
        "prompt": prompt,
        "options": {
            "temperature": 0.0,
            "top_p": 1.0,
            "num_predict": 10,
            "stop": ["\n"]
        }
    }

    response = requests.post("http://localhost:11434/api/generate", json=payload, stream=True)

    if response.status_code == 200:
        result = ""
        for line in response.iter_lines():
            if line:
                data = requests.utils.json.loads(line)
                if "response" in data:
                    result += data["response"]
        return result.strip()
    else:
        return f"Erro: {response.status_code}"

# Worker para inferência paralela
def worker(model, idx, prompt):
    resposta = run_ollama(model, prompt)
    return idx, resposta

# Carrega dataset
dataset = pd.read_csv(INPUT_CSV)

# Loop pelos modelos
for model, run_name in zip(models, run_names):
    print(f"🚀 Iniciando job para modelo: {run_name}")
    start_time = time.time()

    wandb.init(
        project="ollama_eval_mmlu",
        name=f"{run_name}",
        config={
            "model_name": model,
            "run_name": run_name
        }
    )

    df = dataset.copy()
    df["output_model"] = ""

    prompts = df["formatted_prompt"].tolist()

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(worker, model, idx, prompt) for idx, prompt in enumerate(prompts)]

        for i, future in enumerate(tqdm(as_completed(futures), total=len(prompts), desc=f"Processando {run_name}")):
            idx, resposta = future.result()
            df.at[idx, "output_model"] = resposta

            if i % 20 == 0:
                df.to_csv(f"/home/annap/Documents/chatbot_copy/INFERENCE/DATASETS/OLLAMA/mmlu_eval_{run_name}.csv", index=False)

    df.to_csv(f"/home/annap/Documents/chatbot_copy/INFERENCE/DATASETS/OLLAMA/mmlu_eval_{run_name}.csv", index=False)

    elapsed_time = time.time() - start_time
    formatted_time = time.strftime('%H:%M:%S', time.gmtime(elapsed_time))

    wandb.log({
        "elapsed_time_sec": elapsed_time,
        "elapsed_time_hms": formatted_time
    })

    print(f"✅ Modelo {run_name} finalizado em {formatted_time}")

    wandb.finish()

    del df
    gc.collect()

print("🎉 Todos os modelos Ollama processados!")
