from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

model = ChatGroq(
    model="deepseek-r1-distill-llama-70b",
)

class Person(BaseModel):
    name: str = Field(description="name of the person")
    age: int = Field(gt=18, description="Age of the person")
    city: str = Field(description="city name of the person belong to")

parser = PydanticOutputParser(pydantic_object= Person)

template = PromptTemplate(
    template="write a fictional person's name, age, and city name \n {format_instruction}",
    input_variable=[],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)
#print(template)

chain = template | model | parser
result = chain.invoke({})
print(result)