from pydantic import BaseModel, EmailStr, Field
from typing import Optional


class Student(BaseModel):
    name: str = "Shezan"  # default value
    age: Optional[int] = None  # optional value
    cgpa: float = Field(gt=0, lt=4, description="A decimal value for cgpa of a student")  # cgpa should be between 0 and 4
    email: EmailStr


new_student = {
    "name": "Shezan Ahmed",
    # we can validate data type by pydantic which is not possible in normal python or TypeDict
    "age": 23,
    "cgpa": 3.79,
    "email": "abc@gmail.com"  # this will raise an error because it is not a valid email
}

student = Student(**new_student)

print(student)
print(student.dict())
print(type(student))
