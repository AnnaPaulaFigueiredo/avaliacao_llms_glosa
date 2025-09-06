import os
import time
import logging
from datetime import datetime, timedelta
import pandas as pd
from tqdm import tqdm  
import google.generativeai as genai
import pdb

API_KEY_UFLA = ""

INSTRUCTION_FIGURATIVE = """Você é um assistente que vai analisar uma base de dados para identificar figuras de linguagem ou expressões idiomáticas.
Para cada entrada, você deve:
1. Identificar as figuras de linguagem ou expressões idiomáticas na coluna (ORIGINAL).
2. Coloque o trecho que represente a figura de linguagem ou expressão na coluna (FIGURATIVO).
3. Reescrever o trecho identificado de forma literal (sem a figura de linguagem) em (REESCRITO).
4. Substituir o trecho original na frase (ORIGINAL) e coloque em (RESULTADO).
5. Retorne somente um json com os campos ID, ORIGINAL, FIGURATIVO, REESCRITO, RESULTADO."""
 
INSTRUCTION_GLOSA = """Atue como um tradutor de LIBRAS (Língua Brasileira de Sinais).
Textos em português serão enviados e você deve converter para a estrutura de GLOSA, respeitando as seguintes regras:
Estrutura de GLOSA mais comum: Objeto + Sujeito + Verbo (OSV) ou Sujeito + Verbo + Objeto (SVO).

Exemplo de few-shot learning:
ID, PT, GLOSA
305, Você sabe onde fica a loja que vende e aluga apartamentos?, VOCÊ SABER ONDE LOJA CASA APARTAMENTO VENDA ALUGUEL ?
75, Você recebeu o salário em dinheiro? Acautele-se e esconda-o no sapato., VOCÊ RECEBER JÁ SÁLARIO DINHEIRO? OLHAR JÁ ESCONDER SAPATO.
4543	Antes eu ganhava pouco, só recebia bolsa de estudo; agora prosperei, ganho bem.	EU ANTES GANHAR POUCO SÓ RECEBER DINHEIRO BOLSA AGORA EU PROSPERAR GANHAR BEM.
4506	No princípio eu não sabia como fazer o dicionário no computador, agora já acostumei.	PRINCÍPIO EU SABER NADA COMO FAZER DICIONÁRIO COMPUTADOR AGORA JÁ ACOSTUMAR.
131	Aqui no Brasil transforma-se a cana em açúcar. Lá na Europa é diferente, transforma-se a beterraba em açúcar.	BRASIL AQUI CANA TRANSFORMAR AÇÚCAR LÁ EUROPA DIFERENTE BETERRABA TRANSFORMAR AÇÚCAR.
3254	Os animais e os bebês precisam tomar leite, é bom para ajudar contra as doenças.	QUALQUER ANIMAL OU BEBÊ PRECISAR TOMAR LEITE BOM AJUDAR CONTRA DOENÇA.
5394	Aqui no Paraná, em julho, sempre há toró. Prefiro viajar para Fortaleza, onde sempre há sol.	PARANÁ JULHO SEMPRE TORÓ EU PREFERIR VIAJAR FORTALEZA LÁ SEMPRE SOL.
305	Você sabe onde fica a loja que vende e aluga apartamentos?	VOCÊ SABER ONDE LOJA CASA APARTAMENTO VENDA ALUGUEL ?
4459	Eu tenho biscoitos salgados e doces, qual você prefere?	EU TER BISCOITO SAL DOCE VOCÊ PREFERIR QUAL?
2937	Você vai comprar carro!? Impossível! Onde irá conseguir dinheiro?	VOCÊ VAI COMPRAR CARRO!? IMPOSSÍVEL! ONDE CONSEGUIR DINHEIRO?
3477	Qual você prefere com o pão: manteiga ou queijo?	PÃO VOCÊ PREFERIR MANTEIGA OU QUEIJO QUAL?
4970	O sabor desse suco parece de tangerina ou laranja. Qual é?	SUCO SABOR PARECER TANGERINA OU LARANJA QUAL?
4024	Ontem fui à padaria comprar pão e levei um susto com o aumento do preço!	ONTEM EU IR PADARIA COMPRAR PÃO EU SUSTO PREÇO AUMENTAR.
4865	É melhor retirar um pouco o que você escreveu, eu não gostei!	VOCÊ ESCREVER EU NÃO-GOSTAR MELHOR VOCÊ RETIRAR POUCO.
4135	O carro entrou na São Paulo-Santos e paguei o valor do pedágio com aumento!	SÃO-PAULO SANTOS CARRO ENTRAR PAGAR PEDÁGIO VALOR AUMENTAR!
4679	No Rio de Janeiro há poucos rabinos, em São Paulo há mais!	RABINO RIO TER POUCO SÃO-PAULO TER MAIS!
283	Ontem, eu estava com enxaqueca, tomei comprimido, a cabeça melhorou e aliviou!	ONTEM EU ENXAQUECA TOMAR-COMPRIMIDO CABEÇA MELHOR ALIVIAR!

Faça o que se pede:
1. Traduza a frase reescrita PT para GLOSA e insira o resultado na coluna GLOSA.
2. Retorne somente um json com os campos  ID, PT, GLOSA."""

def load_and_clean_data(file_path):

    df = pd.read_csv(file_path)
    print(f"Quantidade de instâncias após carregar e limpar os dados: {len(df)}")
    print(f"Tipo de dados do DataFrame: {type(df)}\n")
    
    return df

def estimate_and_bucket(df, bucket_size=1500): # primeira tentativa 3000, SEGUNDA 1500
    # alterar aqui para quando for glosa e equando for figurative
    # ORIGINAL ou PT
    df['TOTAL_TOKENS_EST'] = df['TOKENS_EST_PT'] + df['TOKENS_EST_ID']
    df['ACUMULADO'] = df['TOTAL_TOKENS_EST'].cumsum()
    df['BUCKET'] = (df['ACUMULADO'] // bucket_size) + 1
    
    print(f"Quantidade de instâncias após estimar tokens e criar buckets: {len(df)}")
    print(f"Tipo de dados do DataFrame após transformação: {type(df)}\n")
    
    return df

def transform_data_to_text(df, col):
    
    #data_to_text = df[['id', 'instance', 'BUCKET']].rename(columns={'id':'ID', 'instance':'ORIGINAL'}).copy()
    #data_text = data_to_text.groupby('BUCKET')[['ID', 'ORIGINAL']].apply(lambda x: x.values.tolist()).tolist()

    data_text = df.groupby('BUCKET')[['ID', col]].apply(lambda x: x.values.tolist()).tolist()

    for sublist in data_text:
        sublist.insert(0, ['ID', col])

    data_text = [str(sublist) for sublist in data_text]
    
    print(f"Quantidade de listas após transformação para texto: {len(data_text)}")
    print(f"Tipo de dados das listas: {type(data_text)}\n")
    
    return data_text

def select_sublists_for_testing(data_text, indices):
    selected_sublists = []
    for i in indices:
        if i < len(data_text):
            selected_sublists.append(data_text[i])
        else:
            print(f"Índice {i} fora do intervalo")
    
    print(f"Quantidade de sublistas selecionadas: {len(selected_sublists)}")
    print(f"Tipo de dados das sublistas selecionadas: {type(selected_sublists)}\n")
    
    return selected_sublists

def save_metadata_to_txt(metadata, index, file_path):
    with open(file_path, 'a') as file:
        file.write(f'bucket {index}\n')
        file.write(str(metadata) + '\n')
    logging.info(f"Metadados do bucket {index} salvos com sucesso.")

def save_cleaned_response_to_txt(response_text, file_path):
    cleaned_response_text = response_text.replace("``` json", "")
    cleaned_response_text = response_text.replace("''' json", "")
    cleaned_response_text = response_text.replace("´´´ json", "")
    cleaned_response_text = cleaned_response_text.strip().strip("'''").strip('"""')
    cleaned_response_text = cleaned_response_text.strip().strip("´´´").strip("´´´")
    cleaned_response_text = cleaned_response_text.strip().strip("```").strip("```")
    with open(file_path, 'a') as file:
        file.write(cleaned_response_text + '\n')
    logging.info("Resposta salva com sucesso no arquivo de texto.")

def instantiate_model(instruction:str):

    model = genai.GenerativeModel(model_name="gemini-1.5-flash",
                                   system_instruction=instruction,)
                                       
    logging.info("Modelo instanciado.")
    return model

def process_data_in_gemini(data_text, INSTRUCTION, response_file_path, metadata_file_path):

    start_time = datetime.now()
    model = instantiate_model(INSTRUCTION)

    for i, data in tqdm(enumerate(data_text), total=len(data_text), desc="Processando buckets"):
        try:
            if datetime.now() - start_time > timedelta(hours=1):
                model = instantiate_model(INSTRUCTION)  
                start_time = datetime.now()
                logging.info(f"Modelo re-instalado após 1h de execução, continuando do bucket {i}.")

            logging.info(f"Iniciando processamento do bucket {i}")
            
            response = model.generate_content(data)
            
            save_cleaned_response_to_txt(response.text, response_file_path)

            if response.usage_metadata:
                save_metadata_to_txt(response.usage_metadata, i, metadata_file_path)
            else:
                logging.warning(f"Bucket {i}: Nenhum metadado de uso encontrado.")
            
        except Exception as e:
            logging.error(f"Ocorreu um erro ao processar o bucket {i}: {e}")
    
    logging.info("Processamento de buckets concluído.")

def process_instance_in_gemini(df, instruction: str, response_file_path, metadata_file_path, column_to_process):
    """Função que envia os dados instância por instância ao Gemini."""
    start_time = datetime.now()
    model = instantiate_model(instruction)

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Processando instâncias"):
        try:
            if datetime.now() - start_time > timedelta(hours=1):
                model = instantiate_model(instruction)
                start_time = datetime.now()
                logging.info(f"Modelo re-instalado após 1h de execução, continuando da instância {i}.")

            logging.info(f"Iniciando processamento da instância {i}")
            
            data = [{'ID': row['ID'], column_to_process: row[column_to_process]}]
            response = model.generate_content(str(data))
            
            save_cleaned_response_to_txt(response.text, response_file_path)

            if response.usage_metadata:
                save_metadata_to_txt(response.usage_metadata, i, metadata_file_path)
            else:
                logging.warning(f"Instância {i}: Nenhum metadado de uso encontrado.")
            
        except Exception as e:
            logging.error(f"Ocorreu um erro ao processar a instância {i}: {e}")
    
    logging.info("Processamento das instâncias concluído.")

def main(process_type, mode, file_name):
    """
    process_type: Define o tipo de processamento ('glosa' ou 'figurative').
    mode: Define o modo de processamento ('bucket' ou 'single').
    """
    logging.basicConfig(filename='../LOGS/'+file_name+'.log', 
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

    genai.configure(api_key=API_KEY_UFLA)
    
    process_start_time = time.time()

    if process_type == "glosa":
        file_path = "../DATASETS/"+file_name+".csv"
        response_file_path = "../DATASETS/"+"glosa_"+file_name+"_response.txt"
        metadata_file_path = "../DATASETS/"+"glosa_"+file_name+"_metadata.txt"
        instruction = INSTRUCTION_GLOSA
        column_to_process = "PT"

    elif process_type == "figurative":
        file_path = "../DATASETS/"+file_name+".csv"
        response_file_path = "../DATASETS/"+"figurative_"+file_name+"_response.txt"
        metadata_file_path = "../DATASETS/"+"figurative_"+file_name+"_metadata.txt"
        instruction = INSTRUCTION_FIGURATIVE
        column_to_process = "ORIGINAL"

    else:
        raise ValueError("O tipo de processamento deve ser 'glosa' ou 'figurative'.")

    df = load_and_clean_data(file_path)
    
    if mode == "bucket":
        df = estimate_and_bucket(df)
        data_text = transform_data_to_text(df, column_to_process)
        process_data_in_gemini(data_text, instruction, response_file_path, metadata_file_path)
    elif mode == "single":
        process_instance_in_gemini(df, instruction, response_file_path, metadata_file_path, column_to_process)
    else:
        raise ValueError("O modo deve ser 'bucket' ou 'single'.")

    total_process_time = time.time() - process_start_time
    logging.info(f"Tempo total de execução: {total_process_time:.2f} segundos.")
    print(f"Tempo total de execução: {total_process_time:.2f} segundos.")

if __name__ == "__main__":
    # Escolha o tipo e o modo de processamento aqui
    # figurative or glosa
    # single or bucket 

    main(process_type="glosa", mode="single", file_name='mmlu_pro_to_gloss_single')
