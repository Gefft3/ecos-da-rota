import pandas as pd
from langchain_community.llms import Ollama
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import DataFrameLoader
import ollama
from tqdm import tqdm
import numpy as np
import sys
import tiktoken
import os
from typing import List
from langchain_core.documents import Document
from langchain_core.runnables import chain


def load_data(url_train, url_test):
    # Datasets definidos pelo shell script
    train = pd.read_csv(url_train)
    test = pd.read_csv(url_test)
    return train, test

def format_text(text):
    text = text.split("$")
    return {"classe": text[0], "justificativa": text[1]}

@chain
def retriever(query: str) -> List[Document]:
    docs, scores = zip(*vectorstore.similarity_search_with_score(query,k=K_MAX))
    for doc, score in zip(docs, scores):
        doc.metadata["score"] = score

    return docs


def ollama_llm(question, context, i, path_outputs):

    prompt_specs = """
Classifique se as questões são Relevantes ou Irrelevantes para o contexto da área de saúde epidemiológica, pense da perspectiva de um profissional de saúde como um médico, mas também como um epidemiologista, enfermeiro, ou outro profissional de saúde.

O formato de saída deve ser o seguinte: classe$justificativa
Sendo que a classe deve ser separada da justificativa por '$' TODAS AS VEZES.

Exemplos:
Questão: 'Tenho dores pulmonares afetadas pelo cigarro.'
Saída esperada: Relevante$O texto trata de um problema respiratório diretamente relacionado à área da saúde.

Questão: 'Quais são os melhores livros de autoajuda?'
Saída esperada: Irrelevante$A questão não se relaciona diretamente com o contexto de saúde ou cuidados médicos.

Faça:
- Classifique a questão como Relevante ou Irrelevante.
- A saída deve ser apenas a classe e a justificativa, separadas por '$', nada mais. 
- Classifique a questão também levando em consideração o contexto fornecido.

Não faça:
- Não adicione informações adicionais à saída.
- Não classifique mais de uma questão, apenas a que foi fornecida.
- Não adicione informações adicionais ao contexto fornecido.
- Não mostre a questão ou o contexto na saída.
"""

    formatted_prompt = f"{prompt_specs}\nQuestão: {question}\n\nContexto: {context}"

    path_arquivo_de_prompts = os.path.join(path_outputs, "prompts.txt")

    with open(path_arquivo_de_prompts, "a") as f:
        f.write(f'Question {i}\n')
        f.write(f'{formatted_prompt}\n')
        f.write("--------------------------------\n\n")

    response = ollama.chat(model='llama3.1', messages=[{'role': 'user', 'content': formatted_prompt}])
    response = response['message']['content']
    return response

    
def rag_chain(question, max_prompt_length, i, path_outputs):    
    retrieved_docs = retriever.invoke(question)

    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens_question = len(encoding.encode(question))
    prompt_tokens = num_tokens_question + 318
    
    formatted_context = ""
    soma_das_distancias = []
    for doc in retrieved_docs:
        if len(encoding.encode(doc.page_content)) + prompt_tokens < max_prompt_length:
            prompt_tokens += len(encoding.encode(doc.page_content))
            formatted_context += "\n\n" + doc.page_content
            soma_das_distancias.append(doc.metadata["score"])

    media_das_distancias = np.mean(soma_das_distancias)

    return ollama_llm(question, formatted_context, i, path_outputs), media_das_distancias

def run_test(df, max_prompt_length, path_outputs):
    
    path_arquivo_de_erros = os.path.join(path_outputs, "Erros.txt")
    path_arquivo_de_classificacoes = os.path.join(path_outputs, "classificacoes.txt")
    path_arquivo_de_distancias = os.path.join(path_outputs, "distancias.txt")

    i = 0
    for text in tqdm(df['text']):
        try:
            response, media_das_distancias = rag_chain(text, max_prompt_length, i, path_outputs) 
            response = format_text(response)
            choice = response['classe']

            if choice == "Relevante" or choice == "Irrelevante":
                with open(path_arquivo_de_classificacoes, "a") as f:
                    f.write(f"{i} {choice}\n")
                
                with open(path_arquivo_de_distancias, "a") as f:
                    f.write(f"{i} {media_das_distancias}\n")


            else:
                with open(path_arquivo_de_erros, "a") as f:
                    f.write(f"Question {i}\n")
                    f.write(f"Texto de entrada: {text}\n\n")
                    f.write(f"Resposta LLM: {response}\n\n")
                    f.write(f"Resposta final: {choice}\n\n")
                    f.write("------------------------------------------------\n\n")
                pass
                
        except Exception as e:
            with open(path_arquivo_de_erros, "a") as f:
                f.write(f"Question {i}\n")
                f.write(f"Texto de entrada: {text}\n\n")
                f.write(f"Erro (exception): {e}\n\n")
                f.write("------------------------------------------------\n\n")
            pass

        i += 1


    

if __name__ == "__main__":
    df_train, df_test = load_data(sys.argv[1], sys.argv[2])
    
    K_MAX = int(sys.argv[3])
    tipo = sys.argv[4]
    max_prompt_length = int(sys.argv[5])

    path_outputs = f'Logs {tipo}/k = {K_MAX}'

    if not os.path.exists(path_outputs):
        os.makedirs(path_outputs)

    #Carregando os documentos de treino
    # loader = DataFrameLoader(df_train, page_content_column="text")
    # docs = loader.load()

    #Criando instanciando modelo de embedding e criando a vectorstore
    embeddings = OllamaEmbeddings(model='nomic-embed-text')
    vectorstore = Chroma(collection_name='v_db', persist_directory="./chroma_db", embedding_function=embeddings)
    
    #Criando o modelo de recuperação
    # retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': k_max})

    run_test(df_test, max_prompt_length, path_outputs)