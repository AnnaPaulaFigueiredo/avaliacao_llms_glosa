import requests
import json
import pandas as pd
import logging
import os
from tqdm import tqdm
from deep_translator import GoogleTranslator
from datetime import datetime


def download_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        logging.error(f"Erro ao baixar o arquivo: {response.status_code}")
        return None


def setup_logging():
    logging.basicConfig(filename='translation_errors.log', level=logging.ERROR)


def translate_to_portuguese(text):
    try:
        translated_text = GoogleTranslator(source='en', target='pt').translate(text)
        return translated_text, "ok"
    except Exception as e:
        logging.error(f"Erro ao traduzir o texto: {text}. Exceção: {e}")
        return "error", "error"


def save_data(new_data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)


def process_translation(df, save_path, save_interval=100):
    new_data = []
    error_count = 0
    file_full_path = os.path.join(save_path, "alpaca_data_translated.json")

    with tqdm(total=len(df), desc="Traduzindo textos", unit="texto") as pbar:
        for idx, (index, row) in enumerate(df.iterrows()):
            new_dict = {
                'original_index': index,
                'instruction': None if row['instruction'] == '' else translate_to_portuguese(row['instruction'])[0],
                'input': None if row['input'] == '' else translate_to_portuguese(row['input'])[0],
                'output': None if row['output'] == '' else translate_to_portuguese(row['output'])[0]
            }

            # Contando erros
            if new_dict['instruction'] == "error" or new_dict['input'] == "error" or new_dict['output'] == "error":
                error_count += 1

            pbar.set_postfix(erros=error_count)
            pbar.update(1)
            new_data.append(new_dict)

            # Salvar a cada `save_interval` iterações
            if (idx + 1) % save_interval == 0:
                save_data(new_data, file_full_path)

    return new_data

def main():
    setup_logging()
    url = 'https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json'
    save_path = "C:/Users/annap/Documents/UFLA/vicuna/DATASETS"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Registro do início da execução
    start_time = datetime.now()
    logging.info(f"Execução iniciada em: {start_time}")

    data = download_data(url)
    if data:
        df = pd.DataFrame(data).astype({'instruction': str, 'input': str, 'output': str})
        translated_data = process_translation(df, save_path)
        save_data(translated_data, os.path.join(save_path, "alpaca_data_translated_final.json"))
    # Registro do fim da execução
    end_time = datetime.now()
    logging.info(f"Execução finalizada em: {end_time}")
    logging.info(f"Duração total da execução: {end_time - start_time}")
if __name__ == "__main__":
    main()
