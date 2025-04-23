from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

model = ChatGroq(
    model="deepseek-r1-distill-llama-70b"
)
parser = StrOutputParser()

loader = TextLoader("../data/langchain project suggestion.txt", encoding="utf-8")
docs = loader.load()

prompt = PromptTemplate(
    template = "what do you think about this texts, What is about? \n {text}",
    input_variables=["text"]
)

chain = prompt | model | parser

result = chain.invoke(docs[0].page_content)
print(result)