import os
import streamlit as st


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


DB_FAISS_PATH="vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def load_llm():
    llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=grok_api_key)
    return llm



def main():
    st.title("Ask Chatbot!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    user_input = st.chat_input("Pass your prompt here")

    if user_input:
        st.chat_message('user').markdown(user_input)
        st.session_state.messages.append({'role':'user', 'content': user_input})

        CUSTOM_PROMPT_TEMPLATE = """
                Use the pieces of information provided in the context to answer user's question.
                If you dont know the answer, just say that you dont know, dont try to make up an answer. 
                Dont provide anything out of the given context

                Context: {context}
                Question: {question}

                Start the answer directly. No small talk please.
                """
        
       
        try: 
            vectorstore=get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")


            retriever=vectorstore.as_retriever(search_kwargs={'k':3})

            prompt_template = set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)
            llm_model = load_llm()
            output_parser=StrOutputParser()

            rag_chain = (
                {"context": retriever,  "question": RunnablePassthrough()}
                | prompt_template
                | llm_model
                | output_parser
            )
            response = rag_chain.invoke(user_input)

            # result=response["result"]
            # source_documents=response["source_documents"]
            # result_to_show=result+"\nSource Docs:\n"+str(source_documents)
            #response="Hi, I am MediBot!"
            st.chat_message('assistant').markdown(response)
            st.session_state.messages.append({'role':'assistant', 'content': response})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()