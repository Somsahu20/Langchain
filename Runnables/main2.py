from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableSequence,
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
    RunnableBranch
)
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint

load_dotenv()

#todo runnable sequence and runnable passthrough

llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash'
)

prompt1 = PromptTemplate(
    template="You are a big comedian like Jim Carrey. Make a wonderful, light-hearted joke on the topic {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='You are a big comedian like jim carrey. Create a stand up comedy monologue on this joke which should be light-hearted:\n {text}',
    input_variables=['text']
)

parser = StrOutputParser()

chain_joke = RunnableSequence(prompt1, llm, parser)
# result = chain_joke.invoke({'topic': 'AI'})

run_pass = RunnablePassthrough()

# chain_result = RunnableParallel(
#     joke=run_pass,
#     standup=RunnableSequence(prompt2, llm, parser)
# )

# res = chain_result.invoke({'text': result})

# print("Joke: ",res['joke'])
# print("Stand up: ", res['standup'])

# # chain = prompt1 | llm | parser | prompt2 | llm | parser
# chain = RunnableSequence(prompt1, llm, parser, prompt2, llm, parser)


# print(result)

#!------------------------------------------------

#todo Runnable parallel

# llm2 = HuggingFaceEndpoint(
#     model="deepseek-ai/DeepSeek-V3", 
#     task="conversational",
# )

# prompt3 = PromptTemplate(
#     template="You are an expert in this {topic} domain. Generate a twitter/x post on {topic}",
#     input_variables=['topic']
# )

# prompt4 = PromptTemplate(
#     template="You are an expert in this {topic} domain. Generate a Linkedin post on {topic}",
#     input_variables=['topic']
# )

# chain = RunnableParallel(
#     linkedin=RunnableSequence(prompt4 | llm | parser),
#     twitter=RunnableSequence(prompt3 | llm | parser) 
# )

# result = chain.invoke({'topic': 'AI'})

# print("LinkedIn: ", result['linkedin'])
# print('Twitter: ', result['twitter'])

#!-------------------------------------------

#todo runnable lambda

def word_counter(text):
    return len(text.split())



# chain_result = RunnableParallel(
#     joke=run_pass,
#     num_words=RunnableLambda(word_counter)
# )

# final_chain = RunnableSequence(chain_joke, chain_result)
# res = final_chain.invoke({'topic': 'AI'})

# print(res['joke'])
# print(res['num_words'])


#!--------------------------------

#todo: Runnable Branch

prompt5 = PromptTemplate(
    template="You are a political expert. Generate a report on the topic {topic}",
    input_variables=['topic']
)

report_chain = RunnableSequence(prompt5, llm, parser)

def count_word(text):
    return len(text.split())

prompt6 = PromptTemplate(
    template="You are a poltical expert. Generate a summary on this text: {text}",
    input_variables=['text']
)

summary_chain = RunnableSequence(prompt6, llm, parser)
as_it_is_chain = RunnablePassthrough()  #? No function passed, passthrough input as-is

condition_chain = RunnableBranch(
    (lambda x: len(x.split()) > 500, summary_chain),
    as_it_is_chain
)

final_chain = RunnableSequence(summary_chain, as_it_is_chain)

res = final_chain.invoke({'topic': 'AI'})

print(res)
    



