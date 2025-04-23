from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
parser = StrOutputParser()
model = ChatGroq(
    model = "deepseek-r1-distill-llama-70b"
)

url = "https://www.geeksforgeeks.org/introduction-to-langchain/"
loader = WebBaseLoader(url)

docs = loader.load()

prompt = PromptTemplate(
    template = "what is langchain and what it can do based on this texts: \n {text}",
    input_variables = ["text"]
)

chain = prompt | model | parser
result = chain.invoke({"text": docs[0].page_content})

print(result)