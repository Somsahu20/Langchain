from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import traceback

load_dotenv()

llm = HuggingFaceEndpoint(
    model="mistralai/Mistral-7B-Instruct-v0.2", 
    task="text-generation",
)

try:
    model = ChatHuggingFace(llm=llm)
    result = model.invoke("When was C# released by Microsoft?")
    print(result.content)
except Exception as err:
    print(f"Error type: {type(err).__name__}")
    print(f"Error message: {err}")
    print("\nFull traceback:")
    traceback.print_exc()