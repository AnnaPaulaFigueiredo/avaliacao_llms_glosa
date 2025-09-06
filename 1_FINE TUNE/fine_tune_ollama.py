import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import torch

def run_finetune(model_name, config, dataset_path):
    output_dir = os.path.join(config["output_base_dir"], model_name.replace(":", "_"))
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[INFO] Carregando modelo: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16
    )

    # Preparar modelo para LoRA
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    print(f"[INFO] Carregando dataset")
    dataset = load_dataset("json", data_files=dataset_path)
    tokenized_dataset = dataset.map(lambda e: tokenizer(e["text"], truncation=True, padding="max_length", max_length=config["max_seq_length"]), batched=True)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        warmup_steps=config["warmup_steps"],
        weight_decay=config["weight_decay"],
        learning_rate=config["learning_rate"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        fp16=True,
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        lr_scheduler_type=config.get("lr_scheduler_type", "linear"),
        report_to="wandb",
        run_name=f"{config['project_name']}-{model_name.replace(':','_')}"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=data_collator
    )

    print(f"[INFO] Iniciando fine-tuning para {model_name}")
    trainer.train()
    model.save_pretrained(output_dir)
    print(f"[INFO] Fine-tuning finalizado para {model_name}")

# Carregar configuração
with open("config_v1_simple.json", "r") as f:
    config = json.load(f)

# Configurar WandB API key
os.environ["WANDB_API_KEY"] = config["wandb_api_key"]

# Dataset a ser utilizado
dataset_path = "./dataset/chatbot_train_data.jsonl"

# Iterar sobre os modelos da config
for model_name in config["models"]:
    run_finetune(model_name, config, dataset_path)

print("\n[INFO] Todos os modelos foram treinados com sucesso.")
