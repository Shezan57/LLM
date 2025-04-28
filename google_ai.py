from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
# from google import genai
import os
import requests
os.environ["GOOGLE_API_KEY"]

prompt = PromptTemplate(
    
    template="You are a helpful assistant. Answer the question \n{input}",
    input_variables=["input"] 
)

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.2)

response = llm.invoke(prompt.invoke({"input": "What is the capital of France?"}))
print(response)