import pandas as pd
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import DataFrameLoader
import ollama
from tqdm import tqdm



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

O formato de saída deve ser o seguinte: classe$justificativa
Sendo que a classe deve ser separada da justificativa por '$' TODAS AS VEZES.

A classe deve ser uma string das categorias acima: 'Relevante' ou 'Irrelevante'.
A justificativa deve ser um texto explicando o motivo da escolha da categoria.

Exemplos:
1. Questão: 'Tenho dores pulmonares afetadas pelo cigarro.'
    Saída: Relevante$O texto trata de um problema respiratório diretamente relacionado à área da saúde.

2. Questão: 'Como posso aumentar minha produtividade no trabalho?'
    Saída: Irrelevante$A questão não está relacionada diretamente ao contexto da área da saúde.

3. Questão: 'Tenho dores nas costas após longas horas sentado.'
    Saída: Relevante$A questão aborda um problema de saúde ocupacional que afeta a coluna vertebral.

4. Questão: 'Quais são os melhores livros de autoajuda?'
    Saída: Irrelevante$A questão não se relaciona diretamente com o contexto de saúde ou cuidados médicos.

Não se esqueça das orientações e forneça a resposta no formato de saída indicado.

A saída deve ser estritamente no formato indicado.

A classificação deve ser exclusivamente entre 'Relevante' e 'Irrelevante' e deve ser apenas uma classificação por questão.
    """
        
    formatted_prompt = f"{prompt_specs}\nQuestão: {question}\n\nContexto: {context}"
        
    #salvar em um arquivo de log para debugar
    with open("logs prompt.txt", "a") as f:
        f.write(f"Questão {i}\n")
        f.write(formatted_prompt)
        f.write("\n")
        f.write("--------------------------------------------------------------------------------------------------------------------")
        f.write("\n")
    

    response = ollama.chat(model='llama3.1', messages=[{'role': 'user', 'content': formatted_prompt}])
    response = response['message']['content']

    return response


def rag_chain(question, i):
    
    retrieved_docs = retriever.invoke(question)
    formatted_context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return ollama_llm(question, formatted_context, i)

def run_test(df):
    
    i = 0
    for text in tqdm(df['text']):
        try:
            response = rag_chain(text, i) 
            
            response = format_text(response)
            
            choice = response['classe']
            
            with open("respostas.txt", "a") as f:
                f.write(f"{response}\n")

            with open("classificacoes.txt", "a") as f:
                f.write(f"{choice}\n")

        except Exception as e:
            print(f"Erro {e}, texto: {text}")
            print(f"Resposta: {response}")
            pass

        if i == 200: break
        i += 1
            
    
df_train = pd.read_csv('./datasets/relevantes/EIOS_train.csv')  
df_test = pd.read_csv('./datasets/relevantes/EIOS_test.csv')

#Carregando os documentos de treino
loader = DataFrameLoader(df_train, page_content_column="text")
docs = loader.load()

#Criando instanciando modelo de embedding e criando a vectorstore
embeddings = OllamaEmbeddings(model='nomic-embed-text')
vectorstore = Chroma(collection_name='v_db', persist_directory="./chroma_db", embedding_function=embeddings)

retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': 1})
run_test(df_test)
# sending_wandb(corrects, incorrects, "Relevante")    