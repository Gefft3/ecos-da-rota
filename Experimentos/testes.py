import pandas as pd
import numpy as np
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import DataFrameLoader
import ollama
import ast
from tqdm import tqdm
import sys
import wandb

#Definindo o modelo a ser usado pela langchain
llm = Ollama(model='llama3.1')

train = {
    "text": ['teste']
}

df_train = pd.DataFrame(train)

#Carregando os documentos de treino
loader = DataFrameLoader(df_train, page_content_column="text")
docs = loader.load()

#Criando instanciando modelo de embedding e criando a vectorstore
embeddings = OllamaEmbeddings(model='nomic-embed-text')
vectorstore = Chroma.from_documents(docs, embedding=embeddings)

retriever = vectorstore.as_retriever(search_type='knn', search_kwargs={'k': 3, 'return_distance': True})
# retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': 3})


# for doc in retriever:
#     print(doc)
    

# # Realizando a consulta e obtendo os documentos retornados
retrievered_docs = retriever.invoke(
    'O Brasil é um país localizado na América do Sul. Possui uma população de 210 milhões de habitantes e é o quinto maior país do mundo em extensão territorial. A capital do Brasil é Brasília e a língua oficial é o português.'
)



# # Iterando sobre os documentos retornados e imprimindo as distâncias (ou similaridade)
print(retrievered_docs)
   
    # print(f"Documento: {doc.page_content}")
    # print(f"Pontuação de similaridade (distância): {doc.metadata}")

