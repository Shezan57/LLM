from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

model = ChatGroq(
    model="deepseek-r1-distill-llama-70b",
)

template1 = PromptTemplate(
    template="write a detailed report on {topic}",
    input_variable=["topic"]
)

template2 = PromptTemplate(
    template="write a 5 line summary on the following text. \n {text}",
    input_variable=["text"]
)

#prompt1 = template1.invoke({"topic": "Black hole"})

# result = model.invoke(prompt1)
#
# prompt2 = template2.invoke({"text": result.content})
# result = model.invoke(prompt2)
# print(result.content)

# We will do same thing with the help of output parser or stroutputparser

parser = StrOutputParser()
chain = template1 | model | parser | template2 | model | parser
result = chain.invoke({"topic": "Black hole"})
print(result)



