from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

# Schema for structured response
class Response(BaseModel):
    news_class: str = Field(description="Classe da resposta", required=True)
    explain: str = Field(description="Justificativa da resposta", required=True)

# Prompt template
prompt = PromptTemplate.from_template(
    """Classifique se as questões são Relevantes ou Irrelevantes para o contexto da área de saúde epidemiológica, pense da perspectiva de um profissional de saúde como um médico, mas também como um epidemiologista, enfermeiro, ou outro profissional de saúde.

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

Questão: {question}
"""
)

llm = ChatOllama(model="llama3.1", format="json", temperature=0)

structured_llm = llm.with_structured_output(Response)

chain = prompt | structured_llm

filled_prompt = prompt.format(question="Minha rua está cheia de buracos")
print(filled_prompt)

question = 'Minha rua está cheia de buracos'

alex = chain.invoke(question)
print(alex.news_class)
print(alex.explain)