from re import template
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

#! Detailed prompt
template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)


#! Summary
template2 = PromptTemplate(
    template='Write a 5 line summary on the text: \n {text}',
    input_variables=['text']
)

# prompt1 = template1.invoke({'topic': 'black hole'})

# result1 = model.invoke(prompt1)

# prompt2 = template2.invoke({'text': result1.content})

# result2 = model.invoke(prompt2)

# print(result2.content)

#!-----------------------

#todo String output parser

# parser = StrOutputParser()
# chain = template1 | model | parser | template2 | model | parser #? chain

# result = chain.invoke({'topic': 'black hole'})

# print(result)

#!------------------

#todo JSON o/p parser

parser = JsonOutputParser()

template3 = PromptTemplate(
    template="Give me the name, age and city of a dc comics superhero {format_instruction}",
    input_variables=[],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

prompt3 = template3.format()

# print(prompt3)

# result = model.invoke(prompt3)
# # print(result)
# final_result = parser.parse(result.content)
# print(type(final_result))
# print(final_result)

chain = template3 | model | parser
result = chain.invoke({}) #! You need to send blank dictinoary as an input parameter as input variables is left empty

print(result)

