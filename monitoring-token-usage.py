from langchain_groq import ChatGroq
from langchain.callbacks import get_openai_callback
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

model = ChatGroq(
    model = 'deepseek-r1-distill-llama-70b'
)

template = PromptTemplate(
    template = "Tel me a joke about {topic}",
    input_variable = ["topic"]
)

chain = template | model
with get_openai_callback() as cb:
    response = chain.invoke({"topic": "football"})
    print(response)
    print(f"Total tokens used: {cb.total_tokens}")
    print(f"Prompt tokens used: {cb.prompt_tokens}")
    print(f"Completion tokens: {cb.completion_tokens}")
    print(f"Total Cost USD: {cb.total_cost}")