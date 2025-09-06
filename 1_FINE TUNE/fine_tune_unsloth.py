
import unsloth
from unsloth import FastLanguageModel, is_bfloat16_supported
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import wandb
import json
import gc
import torch
import subprocess
from functools import partial
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer

def wandb_login(api_key: str):
    wandb.login(key=api_key)

def configure_wandb(project_name, job_type, run_name):
    return wandb.init(project=project_name, job_type=job_type, name=run_name)

def load_model_and_tokenizer(model_name, hf_token, max_seq_length, load_in_4bit=True):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
        token=hf_token,
        device_map="auto",
    )
    return model, tokenizer

def apply_lora(model, lora_config):
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_config["lora_r"],
        target_modules=lora_config["lora_target_modules"],
        lora_alpha=lora_config["lora_alpha"],
        lora_dropout=lora_config["lora_dropout"],
        bias=lora_config["lora_bias"],
        use_gradient_checkpointing=lora_config["use_gradient_checkpointing"],
        random_state=lora_config["random_state"],
        use_rslora=lora_config["use_rslora"],
        loftq_config=lora_config["loftq_config"],
    )
    return model

def load_dataset_from_hugging_face(dataset_name):
    return load_dataset(dataset_name)

def format_chat_template(example, tokenizer):
    instruction = str(example.get("INSTRUCTION", "") or "")
    input_text = str(example.get("INPUT", "") or "")
    output_text = str(example.get("OUTPUT", "") or "")

    prompt = instruction
    if input_text.strip():
        prompt += f" {input_text}"
    prompt += f"\nAssistant: {output_text}"

    return {"text": prompt}


def prepare_dataset(dataset_name, tokenizer, num_proc):
    ds = load_dataset_from_hugging_face(dataset_name)
    ds = ds.map(partial(format_chat_template, tokenizer=tokenizer), num_proc=num_proc)
    return ds

def show_current_memory_stats():
    gpu_stats = torch.cuda.get_device_properties(0)
    used = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    total = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU: {gpu_stats.name}. Used: {used} GB / {total} GB.")
    return used, total

def fine_tune_with_sft(model, tokenizer, dataset, output_dir, train_args, max_seq_length, dataset_num_proc, packing):
    show_current_memory_stats()

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=dataset_num_proc,
        packing=packing,
        args=TrainingArguments(output_dir=output_dir, **train_args)
    )

    trainer_stats = trainer.train(resume_from_checkpoint=False)
    show_current_memory_stats()
    wandb.finish()

    model.config.use_cache = True
    return model, tokenizer

def save_model_and_tokenizer(model, tokenizer, repo_name, hf_token):
    try:
        print("🔼 Subindo modelo e tokenizer para Hugging Face Hub...")
        model.push_to_hub(repo_name, use_temp_dir=True, token=hf_token)
        tokenizer.push_to_hub(repo_name, use_temp_dir=True, token=hf_token)
        print("✅ Upload completo.")
    except Exception as e:
        print(f"❌ Erro no upload: {e}")

def run_fine_tuning_pipeline(config_path):
    from huggingface_hub import login
    login(token="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")
    with open(config_path) as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📌 Dispositivo: {device}")

    wandb_login(config["wandb_api_key"])

    for item in config["models"]:
        model_name = item["model_name"]
        dataset_name = item["dataset_name"]
        repo_name = item["repo_name"]

        print(f"\n🚀 Treinando modelo: {model_name}")

        run = configure_wandb(config["project_name"], "training", repo_name)

        model, tokenizer = load_model_and_tokenizer(
            model_name,
            config["hf_read_token"],
            config["training_args"]["max_seq_length"],
        )

        model = apply_lora(model, config)

        dataset = prepare_dataset(dataset_name, tokenizer, config["training_args"]["dataset_num_proc"])

        torch.cuda.empty_cache()

        train_args = {k: v for k, v in config["training_args"].items() if k not in ["max_seq_length", "dataset_num_proc", "packing"]}
        model, tokenizer = fine_tune_with_sft(
            model,
            tokenizer,
            dataset,
            f"{config['output_base_dir']}/{repo_name}",
            train_args=train_args,
            max_seq_length=config["training_args"]["max_seq_length"],
            dataset_num_proc=config["training_args"]["dataset_num_proc"],
            packing=config["training_args"]["packing"],
        )

        save_model_and_tokenizer(model, tokenizer, repo_name, config["hf_write_token"])

        del model, tokenizer, dataset
        torch.cuda.empty_cache()
        gc.collect()

        print(f"✅ Finalizado: {model_name}\n")

        run.finish()

    print("🎉 Todos os treinamentos concluídos.")

if __name__ == "__main__":
    run_fine_tuning_pipeline("config_1.json")
