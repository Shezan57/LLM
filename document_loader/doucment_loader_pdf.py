from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

model = ChatGroq(
    model="deepseek-r1-distill-llama-70b"
)

parser = StrOutputParser()

loader = PyPDFLoader("../data/The Role of Artificial Intelligence in Revolutionizing Construction Project Management Enhancing Efficiency and Sustainability.pdf")

docs = loader.load()

prompt = PromptTemplate(
    template = "what is this pdf file about ? \n {text}",
    input_variables=["text"]
)

chain = prompt | model | parser

result = chain.invoke(docs[0].page_content)
print(result)
