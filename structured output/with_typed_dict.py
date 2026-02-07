from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, Annotated, List
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# class Response(TypedDict):
#     summary: str
#     sentiment: str

class Response(BaseModel):
    key_themes: Annotated[List[str], "Write down all the key themes discussed in the review in a list"]
    summary: Annotated[str, "A brief description of summary"]
    sentiment: Annotated[str, "Return sentiment of the review"] 
    pros: Annotated[List[str], "Write down all the positives"]
    cons: Annotated[List[str], "Write down all the negatives"]


model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

structured_model = model.with_structured_output(Response)

result = structured_model.invoke("""
    The car is not great. The fit and finish of this car is what drags it down. Though the enigne is top notch, produing 450 nm of torque at 170 ps of power
""")

print(result.summary)
