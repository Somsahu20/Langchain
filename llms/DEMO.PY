from langchain_google_genai import ChatGoogleGenerativeAI as gc
from dotenv import load_dotenv

load_dotenv()

llm = gc(model="gemini-2.5-flash", temperature=0.2)

res = llm.invoke("What is the capital of libya?")#! Takes input as string and gives output as string

print(res.content)