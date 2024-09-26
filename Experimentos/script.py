import pandas as pd
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import DataFrameLoader
import ollama
from tqdm import tqdm
import numpy as np
import sys
import tiktoken


def load_data(url_train, url_test):
    # Datasets definidos pelo shell script
    train = pd.read_csv(url_train)
    test = pd.read_csv(url_test)
    return train, test

def format_text(text):
    text = text.split("$")
    return {"classe": text[0], "justificativa": text[1]}


def ollama_llm(question, context, i):

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

    with open("prompts.txt", "a") as f:
        f.write(f'Question {i}\n')
        f.write(f'{formatted_prompt}\n')
        f.write("--------------------------------\n\n")

    response = ollama.chat(model='llama3.1', messages=[{'role': 'user', 'content': formatted_prompt}])
    response = response['message']['content']
    return response

    
def rag_chain(question, retriever, max_prompt_length, i):    
    retrieved_docs = retriever.invoke(question)

    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens_question = len(encoding.encode(question))
    prompt_tokens = num_tokens_question + 318
    
    formatted_context = ""
    for doc in retrieved_docs:
        if len(encoding.encode(doc.page_content)) + prompt_tokens < max_prompt_length:
            prompt_tokens += len(encoding.encode(doc.page_content))
            formatted_context += "\n\n" + doc.page_content

    return ollama_llm(question, formatted_context, i)

def run_test(df, retriever, tipo, max_prompt_length):
    
    i = 0
    for text in tqdm(df['text']):
        try:
            response = rag_chain(text, retriever, max_prompt_length, i) 
            response = format_text(response)
            choice = response['classe']

            if choice == "Relevante" or choice == "Irrelevante":
                with open(f"classificacoes {tipo}.txt", "a") as f:
                    f.write(f"{i} {choice}\n")
            else:
                with open(f"Erros {tipo}.txt", "a") as f:
                    f.write(f"Question {i}\n")
                    f.write(f"Texto de entrada: {text}\n\n")
                    f.write(f"Resposta LLM: {response}\n\n")
                    f.write(f"Resposta final: {choice}\n\n")
                    f.write("------------------------------------------------\n\n")
                pass
                
        except Exception as e:
            with open(f"Erros {tipo}.txt", "a") as f:
                f.write(f"Question {i}\n")
                f.write(f"Texto de entrada: {text}\n\n")
                f.write(f"Erro (exception): {e}\n\n")
                f.write("------------------------------------------------\n\n")
            pass

        i += 1

def main():
    df_train, df_test = load_data(sys.argv[1], sys.argv[2])
    
    k_max = int(sys.argv[3])
    tipo = sys.argv[4]
    max_prompt_length = int(sys.argv[5])

    #Carregando os documentos de treino
    loader = DataFrameLoader(df_train, page_content_column="text")
    docs = loader.load()

    #Criando instanciando modelo de embedding e criando a vectorstore
    embeddings = OllamaEmbeddings(model='nomic-embed-text')
    vectorstore = Chroma(collection_name='v_db', persist_directory="./chroma_db", embedding_function=embeddings)
    
    #Criando o modelo de recuperação
    retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': k_max})
    run_test(df_test, retriever, tipo, max_prompt_length)

if __name__ == "__main__":
    main()