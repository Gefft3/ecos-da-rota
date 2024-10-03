from pydantic import BaseModel
from langchain.llms import Ollama
from outlines import models, generate

class User(BaseModel):
    name: str
    last_name: str
    id: int


# model = models.transformers("microsoft/Phi-3-mini-4k-instruct")
# # model = models.transformers("meta-llama/Llama-3.1-8B-Instruct")

# generator = generate.json(model, User)
# result = generator(
#     "Create a user profile with the fields name, last_name and id"
# )
# print(result)
# User(name="John", last_name="Doe", id=11)

llm = Ollama(model = 'llama3.1')

# generator = generate.json(llm, User)
# result = generator(
#     "Create a user profile with the fields name, last_name and id"
# )

result = llm.invoke("Create a user profile with the fields name, last_name and id")

print(result)