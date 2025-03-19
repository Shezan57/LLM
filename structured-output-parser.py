from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

model = ChatGroq(
    model="deepseek-r1-distill-llama-70b",
)
schema = [
    ResponseSchema(name="fact_1", description="The first fact about the topic"),
    ResponseSchema(name="fact_2", description="The second fact about the topic"),
    ResponseSchema(name="fact_3", description="The third fact about the topic"),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template="Write three fact about the topic {topic} \n {format_instruction}",
    input_vatiables=["topic"],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

chain = template | model | parser
result = chain.invoke({"topic": "elephants"})
print(result)
