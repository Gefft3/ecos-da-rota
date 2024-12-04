import pandas as pd 
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.runnables import chain
from bs4 import BeautifulSoup
import requests
from pydantic import BaseModel, Field
from tqdm import tqdm
import gc

class Response(BaseModel):
    summary: str = Field(description="Resumo da notícia fornecida.", required=True)

def fetch_page_content(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text(separator="\n").strip()
        return text if text else "Conteúdo vazio ou indisponível."
    except Exception as e:
        return f"Erro ao acessar o link: {e}"
    
def summarize_web_content(url, summary_chain):
    page_content = fetch_page_content(url)
    
    if "Erro" in page_content or "Conteúdo vazio" in page_content:
        return page_content  
    
    summary = summary_chain.invoke(page_content)
    return summary

def model_config():

    summary_prompt = PromptTemplate(
    input_variables=["content"],
    template="""
Você é um assistente especializado em gerar resumos detalhados.

Seu objetivo é fornecer um resumo claro e completo, destacando os principais pontos, argumentos e informações relevantes do conteúdo. O resumo deve ser informativo e bem estruturado, com foco nas ideias principais. 

Lembre-se de que o resumo deve ser coeso e coerente, respeitando um limite máximo de 350 palavras, para permitir um pouco mais de profundidade e detalhes.

Texto:
{content}
"""
)

    llm = ChatOllama(model="llama3.1", format="json", temperature=0.1)

    structured_llm = llm.with_structured_output(Response)

    summary_chain = summary_prompt | structured_llm
    
    return summary_chain

def main():

    url_dataset = '../datasets/irrelevantes/dataset_links.csv'

    dataset = pd.read_csv(url_dataset)

    dataset_output = pd.DataFrame(columns=['link', 'summary'])

    summary_chain = model_config()

    for url in tqdm(dataset['link']):
        gc.collect()
        try:
            response = summarize_web_content(url, summary_chain)
            df_aux = pd.DataFrame({'link': [url], 'summary': [response.summary]})

            dataset_output = pd.concat([dataset_output, df_aux])
            dataset_output.to_csv('../datasets/irrelevantes/dataset_links_summary.csv', index=False)
        
        except Exception as e:
            print(f'Erro: {e}')


if __name__ == "__main__":
    main()