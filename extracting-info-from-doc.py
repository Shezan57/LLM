from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import Optional, List
from langchain.chains import create_extraction_chain_pydantic
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model='deepseek-r1-distill-llama-70b'
)


class Experience(BaseModel):
    start_date: Optional[str] = Field(description="The start date of the experience")
    end_date: Optional[str] = Field(description="The end date of the experience")
    description: Optional[str] = Field(description="The description of the experience")


class Study(Experience):
    degree: Optional[str] = Field(description="The degree of the candidate")
    university: Optional[str] = Field(description="Name of the university of the candidate")
    country: Optional[str] = Field(description="name of the country of the candidate")
    grade: Optional[str] = Field(description="The grade of the candidate")


class WorkExperience(Experience):
    company: Optional[str] = Field(description="The company of the candidate")
    job_title: Optional[str] = Field(description="The job title of the candidate")


class Resume(BaseModel):
    first_name: str = Field(description="The first name of the candidate")
    last_name: str = Field(description="The last name of the candidate")
    linkedin_url: Optional[str] = Field(description="The linkedin url of the candidate")
    email_address: Optional[str] = Field(description="The email address of the candidate")
    nationality: Optional[str] = Field(description="Nationality of the candidate")
    phone_number: Optional[str] = Field(description="The phone number of the candidate")
    skill: Optional[List[str]] = Field(description="The skills of the candidate")
    study: Optional[Study] = Field(description="The studies of the candidate")
    # Change to List of WorkExperience
    work_experience: Optional[List[WorkExperience]] = Field(description="The work experiences of the candidate")
    hobby: Optional[List[str]] = Field(description="The hobbies of the candidate")


pdf_file_path = r"C:\Users\arafa\OneDrive\Desktop\resume.pdf"
pdf_loader = PyPDFLoader(pdf_file_path)
docs = pdf_loader.load_and_split()
resume_text = "\n".join([doc.page_content for doc in docs])

extraction_model = model.with_structured_output(Resume)
result = extraction_model.invoke(resume_text)
print(result)
