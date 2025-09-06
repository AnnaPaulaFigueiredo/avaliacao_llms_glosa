from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import logging
from deep_translator import GoogleTranslator
from datetime import datetime
import warnings
import json
import numpy as np
import re

tqdm.pandas()
warnings.filterwarnings('ignore')

def translate_to_portuguese(text, question_id=None):
    if not text or text.strip() == "":
        return None, "ok"
    try:
        translated_text = GoogleTranslator(source='en', target='pt').translate(text)
        return translated_text, "ok"
    except Exception as e:
        if question_id:
            logging.error(f"Erro ao traduzir o texto para question_id {question_id}: {text}. Exceção: {e}")
        else:
            logging.error(f"Erro ao traduzir o texto: {text}. Exceção: {e}")
        return None, "erro"

def code_detector(text: str) -> bool:
    code_pattern = r"""(
        \bdef\b|
        \bclass\b|
        \belse\b|
        \breturn\b|
        \bprint\b|
        \bimport\b|
        \blambda\b|
        \belif\b|
        \basync\b|
        \bfunction\b|
        \bconst\b|
        \bvar\b|
        \bstatic\b|
        \bvoid\b|
        \bloop\b|
        \benum\b|
        \bimpl\b|
        \bcout|
        \bcin|
        \bnamespace\b|
        std|
        printf|
        System\.Out\.Print|
        System\.Out\.Println|
        \++|
        \--|
        \+=|
        \-=|
        \*=|
        \/=|
        \%=|
        \==|
        \===|
        \!=|
        \!==|
        \>=|
        \<=|
        \&&|
        \||
    )"""
    return bool(re.search(code_pattern, str(text)))

def contains_image_or_link(text):
    image_link_pattern = r"\b\w+\.(png|jpg|jpeg|com)\b"
    return bool(re.search(image_link_pattern, text, re.IGNORECASE))

def is_all_numeric(text):
    text = text.strip()
    if text.startswith('[') and text.endswith(']'):
        try:
            eval_list = eval(text)
            if isinstance(eval_list, list):
                for item in eval_list:
                    if not is_all_numeric(str(item)):
                        return False
                return True
        except:
            return False
    if re.search(r'[a-zA-Z]{2,}', text):
        return False
    math_variable_pattern = r'^[\d\s\+\-\*/\(\)\.,=a-zA-Z]*$'
    if re.match(math_variable_pattern, text):
        return True
    return False

def contains_only_specials_or_emojis(text):
    return bool(re.match(r'^[^\w\s]+$', text))

def contains_image_or_blob(message):
    return any(keyword in message for keyword in ['IPython.display.Image'])

def main(ds_name, dataset_name, dataset_path, columns_to_translate, save_path):
    start_time = datetime.now()
    logging.info(f"Execução iniciada em: {start_time}")

    if dataset_name is not None:
        dataset = load_dataset(dataset_name, split="all")
        df = pd.DataFrame(dataset)
    else:
        df = pd.read_csv(dataset_path)

    logging.info(f"Dataset carregado: {ds_name} com {len(df)} linhas.")
    df.info()

    if 'question_id' not in df.columns:
        raise ValueError("A coluna 'question_id' não está presente no DataFrame.")

    # Verificar duplicação no dataset original
    if df['question_id'].duplicated().any():
        logging.warning("Existem IDs duplicados no dataset original. Removendo duplicados.")
        df = df.drop_duplicates(subset=['question_id'])

    new_data = []
    existing_ids = set()

    for i, (index, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Traduzindo textos")):
        if row['question_id'] in existing_ids:
            logging.error(f"ID duplicado detectado: {row['question_id']}. Pulando...")
            continue
        existing_ids.add(row['question_id'])

        translated_row = {"question_id": row['question_id']}
        error_flag = False

        for col in columns_to_translate:
            if col in df.columns:
                value = row[col]

                if isinstance(value, (list, pd.Series, np.ndarray)):
                    value = " ".join(map(str, value))

                if pd.notna(value):
                    if (code_detector(value) or contains_image_or_link(value) or
                        is_all_numeric(value) or
                        contains_only_specials_or_emojis(value) or contains_image_or_blob(value)):
                        translated_row[col] = value  # Manter o dado original
                        translated_row[f"{col}_status"] = "skipped"
                    else:
                        translated_text, status = translate_to_portuguese(str(value), question_id=row['question_id'])
                        translated_row[col] = translated_text
                        translated_row[f"{col}_status"] = status

                        if status == "erro":
                            error_flag = True
                else:
                    translated_row[col] = None
                    translated_row[f"{col}_status"] = "ok"
            else:
                logging.warning(f"Coluna {col} não encontrada no DataFrame.")
                translated_row[col] = None
                translated_row[f"{col}_status"] = "ok"

        if error_flag:
            logging.error(f"Erro no item ID: {row['question_id']}")

        new_data.append(translated_row)

        if (i + 1) % 100 == 0:
            partial_save_path = f"{save_path.split('.json')[0]}_partial.json"
            with open(partial_save_path, 'w', encoding='utf-8') as f:
                json.dump(new_data, f, ensure_ascii=False, indent=4)

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)

    end_time = datetime.now()
    logging.info(f"Execução finalizada em: {end_time}")
    logging.info(f"Duração total da execução: {end_time - start_time}")

if __name__ == "__main__":
    DS_NAME = "mmlu"

    logging.basicConfig(
        filename='../LOGS/log_translation_' + DS_NAME + '.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    DATASET_H_NAME = None  # from Hugging Face if necessary
    DATASET_PATH = '../DATASETS/' + DS_NAME + '.csv'
    COLUMNS_TO_TRANSLATE = ['question', 'choice_1', 'choice_2', 'choice_3', 'choice_4']
    SAVE_PATH = "../DATASETS/" + DS_NAME + ".json"

    main(DS_NAME, DATASET_H_NAME, DATASET_PATH, COLUMNS_TO_TRANSLATE, SAVE_PATH)