
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough


# Step 1: Setup LLM (Mistral with HuggingFace)
# dotenv ko use karenge environment variables load karne ke liye
import os
from dotenv import load_dotenv

load_dotenv()  # Environment variables load karte hain

# Grok API Key
grok_api_key = os.getenv("GROK_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
# model = 

def load_llm():
    llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=grok_api_key)
    return llm

# Step 2: Connect LLM with FAISS and Create chain

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Load Database
DB_FAISS_PATH="vectorstore/db_faiss"
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)   # yaha wapas local se db load karne ka code likhna hai
retriever=db.as_retriever(search_kwargs={'k':3})


prompt = set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)
llm_model = load_llm()
output_parser=StrOutputParser()
# Create QA chain
rag_chain = (
    {"context": retriever,  "question": RunnablePassthrough()}
    | prompt
    | llm_model
    | output_parser
)





# Now invoke with a single query
user_query=input("Write Query Here: ")
response=rag_chain.invoke(user_query)
print(response)
# print("RESULT: ", response["result"])
# print("SOURCE DOCUMENTS: ", response["source_documents"])