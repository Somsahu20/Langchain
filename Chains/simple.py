from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from pathlib import Path
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

BASE_DIR = Path(__file__).resolve().parent.parent
ENV = BASE_DIR / ".env"

print(ENV)

load_dotenv()


prompt = PromptTemplate(
    template="Generate 5 interesting facts about the topic about {topic}",
    input_variables=['topic']
)

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

# chain = prompt | model | parser

# result = chain.invoke({'topic': 'Bluetooth'})

# print(result)


#!---------------------------------------------------


#todo Sequential Output

# prompt1 = PromptTemplate(
#     template="Generate a detailed summary of the topic: {topic}",
#     input_variables=['topic']
# )

# prompt2 = PromptTemplate(
#     template="Generate a 3 line summary of the text \n {text}",
#     input_variables=['text']
# )

# chain = prompt1 | model | parser | prompt2 | model | parser

# result2 = chain.invoke({'topic': 'GDP of India'})

# print(result2)

#!------------------------------------------------------


#todo parallel chains

model2 = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

# prompt1 = PromptTemplate(
#     template="Generate a summary about the text {text}",
#     input_variables=['text']
# )

# prompt2 = PromptTemplate(
#     template="Generate 5 short questions for the text \n {text}",
#     input_variables=['text']
# )

# prompt3 = PromptTemplate(
#     template="Merge the provided notes and quiz into a single documents \n notes -> {notes} and quiz -> {quiz}",
#     input_variables=['notes', 'quiz']
# )

# parallel_chain = RunnableParallel({
#     'notes': prompt1 | model | parser,
#     'quiz': prompt2 | model2 | parser     #! Parallel chain
# })

# merge_chain = prompt3 | model2 | parser

# chain = parallel_chain | merge_chain

# text = """

# Here’s a short, engaging write-up about cricket that captures its spirit and culture:  

# ***

# Cricket is more than just a sport—it’s a blend of strategy, skill, and passion that connects millions across the world. Played in formats ranging from five-day Test matches to fast-paced T20s, cricket demands a unique balance of patience, precision, and quick thinking. The game revolves around two teams competing to score runs while mastering bowling, batting, and fielding under varying conditions.  

# Beyond the boundaries of the pitch, cricket carries a deep cultural resonance. In countries like India, England, Australia, and Pakistan, it’s almost a way of life—uniting fans in joy, suspense, and pride. Legendary players such as Sachin Tendulkar, Ricky Ponting, and Ben Stokes have turned matches into memorable chapters of sporting history. From backyard games to packed stadiums roaring with chants, cricket’s magic lies in its unpredictability and timeless charm.  

# ***

# Would you like me to make it sound more informative (like for a school essay) or more passionate (like for a sports article)?
# """

# result = chain.invoke({'text': text})

# print(result)


class Response(BaseModel):
    ans: Literal["positive", "sensitive"] = Field(description="Give the sentiment of the feedback")

pyParser = PydanticOutputParser(pydantic_object=Response)

prompt1 = PromptTemplate(
    template="Classify the sentinment of the following feedback based on positive and negative \n {text} \n {format_instructions}",
    input_variables=['text'],
    partial_variables={'format_instructions': pyParser.get_format_instructions()}
)



classify_chain = prompt1 | model2 | pyParser
text = """
I am beyond impressed with this purchase! The build quality feels premium and sturdy, and it performs exactly as advertised—if not better. It’s rare to find a product that balances sleek design with such high efficiency. It has genuinely made my daily routine much easier, and I’d recommend it to anyone looking for a reliable, high-performance option in this category.
"""

prompt2 = PromptTemplate(
    template="Write an appropriate response to this positive feedback \n {feedback}",
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template="Write an appropriate response to this negative feedback \n {feedback}",
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x:x.ans == "positive", prompt2 | model2 | parser),
    (lambda x:x.ans == "negative", prompt3 | model2 | parser),
    RunnableLambda(lambda x: "couldn\'t find any sentiment")
)

chain = classify_chain | branch_chain
print(chain.invoke({'text': text}))