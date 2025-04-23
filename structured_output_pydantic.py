from groq import Groq
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
load_dotenv()
import os

client = Groq()

# Define a prompt template
prompt_template = PromptTemplate(
    input_variables=["text"],
    template="Translate the following English text to French: '{text}'"
)

# Create a LangChain LLMChain with the Groq model
llm_chain = LLMChain(
    llm=client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "Translate the following English text to French: '{text}'"}]
    ),
    prompt=prompt_template
)

# Use the chain to generate a completion
response = llm_chain.run(text="Hello, how are you?")
print(response)