from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

model = ChatGroq(
    model="deepseek-r1-distill-llama-70b",
)

parser = JsonOutputParser()

template = PromptTemplate(
    template="write a fictional person's name, age, and city name \n {format_instruction}",
    input_variable=[],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({})
print(result)
