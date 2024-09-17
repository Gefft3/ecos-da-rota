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

def load_data(url_train, url_test):
    # Datasets definidos pelo shell script
    train = pd.read_csv(url_train)
    test = pd.read_csv(url_test)
    return train, test


def ollama_llm(question, context):

    prompt_specs = """
        Classifique se as questões são relevantes ou irrelevantes para o contexto da área de saúde epidemiológica.
        Pense da perspectiva de um profissional de saúde como um médico, mas também como um epidemiologista, enfermeiro, ou outro profissional de saúde.

        Você deve classificar cada questão em uma das 2 categorias:
        1. Relevante
        2. Irrelevante

        Para cada questão, forneça uma justificativa detalhada para a escolha feita.
        Certifique-se de que a justificativa seja clara e esteja relacionada ao contexto da área da saúde.

        Faça:
        1. Seja específico e claro.
        2. Compreenda o contexto da questão relacionada à saúde antes de categorizá-la.
        3. Aja como um profissional de saúde.

        NÃO faça:
        1. Não adivinhe ou invente informações.
        2. Não crie novas categorias de justificativa; utilize apenas as fornecidas acima.

        Dica: Classifique como irrelevante quaisquer questões que não estejam diretamente relacionadas à saúde ou cuidados médicos.

        O formato de saída deve ser o seguinte:
        {
            "classe": "",
            "justificativa": ""
        }

        A classe deve ser uma string das categorias acima: 'Relevante' ou 'Irrelevante'.
        A justificativa deve ser um texto explicando o motivo da escolha da categoria.

        Exemplos:
        1. Questão: 'Tenho dores pulmonares afetadas pelo cigarro.'
            Saída: { 'classe': 'Relevante', 'justificativa': 'O texto trata de um problema respiratório diretamente relacionado à área da saúde.' }

        2. Questão: 'Como posso aumentar minha produtividade no trabalho?'
            Saída: { 'classe': 'Irrelevante', 'justificativa': 'A questão não está relacionada diretamente ao contexto da área da saúde.' }

        3. Questão: 'Tenho dores nas costas após longas horas sentado.'
            Saída: { 'classe': 'Relevante', 'justificativa': 'A questão aborda um problema de saúde ocupacional que afeta a coluna vertebral.' }

        4. Questão: 'Quais são os melhores livros de autoajuda?'
            Saída: { 'classe': 'Irrelevante', 'justificativa': 'A questão não se relaciona diretamente com o contexto de saúde ou cuidados médicos.' }
        
        Não se esqueça das orientações e forneça a resposta no formato de saída indicado.

        A saída deve ser estritamente no formato indicado, com aspas simples e chaves.

        A classificação deve ser exclusivamente entre 'Relevante' e 'Irrelevante'.
    """
    
    formatted_prompt = f"{prompt_specs}\nQuestão: {question}\n\nContexto: {context}"
    response = ollama.chat(model='llama3.1', messages=[{'role': 'user', 'content': formatted_prompt}])
    return response['message']['content']

def rag_chain(question, retriever):
    retrieved_docs = retriever.invoke(question)
    formatted_context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    # distances = [doc.distance for doc in retrieved_docs]
    # mean_distance = np.mean(distances)
    return ollama_llm(question, formatted_context)

def run_test(df,retriever):
    
    correct = 0
    incorrect = 0
    # distance = 0
    # distance_list = []
    
    for _, row in tqdm(df.iterrows()):
        text = row['text']
        try:
            response = rag_chain(text, retriever)
            # print(response)
            # distance_list.append(distance)
            response = ast.literal_eval(response)
            choice = response['classe'].lower()
            if choice == 'relevante':
                correct += 1
            else:
                incorrect += 1
        except Exception as e:
            flag = True
            print(f"Erro {e} na linha: {_}")
            break
    
    # mean_distance = np.mean(distance_list) 
    print(flag)
    return correct, incorrect, flag

def sending_wandb(corrects, incorrects, tipo):
    accuracy = corrects / (corrects + incorrects)
    wandb.log({f"accuracy/recall - {tipo}": accuracy})
    wandb.finish()


if __name__ == "__main__":

    run = wandb.init(
        project="ECOS da Rota",
    )

    path_train = sys.argv[1]
    path_test = sys.argv[2]
    k_max = int(sys.argv[3])
    tipo = sys.argv[4]

    flag = False

    df_train, df_test = load_data(path_train, path_test)

    #Definindo o modelo a ser usado pela langchain
    llm = Ollama(model='llama3.1')

    #Carregando os documentos de treino
    loader = DataFrameLoader(df_test, page_content_column="text")
    docs = loader.load()

    #Criando instanciando modelo de embedding e criando a vectorstore
    embeddings = OllamaEmbeddings(model='nomic-embed-text')
    vectorstore = Chroma.from_documents(docs, embedding=embeddings)

    #Instanciando o retriever e rodando o teste
    for k in range(1, k_max+1):
        retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': 3})
        corrects, incorrects, flag = run_test(df_test,retriever)
        if flag: break
        sending_wandb(corrects, incorrects, tipo)


    
