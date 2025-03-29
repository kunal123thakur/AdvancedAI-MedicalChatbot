
# Ask Chatbot - README



https://github.com/user-attachments/assets/5fbf0d76-37c0-4687-ac4c-34b43624588a


## Overview
This is a **Streamlit-based chatbot** that leverages **LangChain, FAISS, and Groq LLMs** to provide intelligent responses based on vector search. The chatbot retrieves relevant context using **FAISS vector search** and generates responses using **ChatGroq (Gemma2-9b-It model)**. The pipeline is explained step by step for better understanding.

---

## 📌 Pipeline Explanation

### 1️⃣ **Environment Setup**
- The script loads necessary libraries like **os, streamlit, dotenv, LangChain modules, and HuggingFace embeddings**.
- Loads **environment variables** from `.env` file using `dotenv`.
  
  ```python
  from dotenv import load_dotenv
  load_dotenv()
  
  grok_api_key = os.getenv("GROK_API_KEY")
  HF_TOKEN = os.getenv("HF_TOKEN")
  ```
  
  This ensures **API keys are securely loaded**.

---
![alt text](<Screenshot 2025-03-27 121907.png>)

### 2️⃣ **Vector Database Setup (FAISS)**
- Uses **FAISS** to load a prebuilt vector store from `vectorstore/db_faiss`.
- Embedding model: **`sentence-transformers/all-MiniLM-L6-v2`**.

  ```python
  @st.cache_resource
  def get_vectorstore():
      embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
      db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
      return db
  ```

  This function **caches** the vector store for faster retrieval and reduces redundant computations.

---

### 3️⃣ **Custom Prompt Template**
- A **custom prompt template** is used to structure how the chatbot interacts with the user.
- It prevents hallucinations by ensuring responses **only use retrieved context**.
  
  ```python
  def set_custom_prompt(custom_prompt_template):
      prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
      return prompt
  ```

  Example Template:
  ```
  Use the pieces of information provided in the context to answer user's question.
  If you don't know the answer, just say that you don't know. Don't try to make up an answer.
  Don't provide anything out of the given context.
  ```

---

### 4️⃣ **Loading the LLM (Groq Chat with Gemma2-9b-It)**
- This function initializes the **ChatGroq LLM** using the **Gemma2-9b-It** model.
  
  ```python
  def load_llm():
      llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=grok_api_key)
      return llm
  ```

  This allows the chatbot to generate responses based on retrieved information.

---

### 5️⃣ **Streamlit UI & Chat Interface**
- **Streamlit is used to create the chatbot UI.**
- Messages are stored in `st.session_state.messages` to maintain conversation history.
- `st.chat_input()` captures user input dynamically.
  
  ```python
  if 'messages' not in st.session_state:
      st.session_state.messages = []
  
  for message in st.session_state.messages:
      st.chat_message(message['role']).markdown(message['content'])
  
  user_input = st.chat_input("Pass your prompt here")
  ```

  This ensures **chat history persists across interactions**.

---

### 6️⃣ **Retrieval-Augmented Generation (RAG) Pipeline**
- The **retriever** extracts relevant information from FAISS.
- The **prompt template** structures the input for LLM.
- The **LLM processes the input and generates a response**.
- The **output parser** formats the final response.
  
  ```python
  retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
  
  prompt_template = set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)
  llm_model = load_llm()
  output_parser = StrOutputParser()
  
  rag_chain = (
      {"context": retriever, "question": RunnablePassthrough()}
      | prompt_template
      | llm_model
      | output_parser
  )
  response = rag_chain.invoke(user_input)
  ```

  This **end-to-end RAG pipeline** ensures:
  - **Context retrieval** from FAISS
  - **LLM response generation**
  - **Structured output formatting**

---

### 7️⃣ **Displaying the Assistant's Response**
- The chatbot **displays the assistant's response** in the Streamlit chat interface.
  
  ```python
  st.chat_message('assistant').markdown(response)
  st.session_state.messages.append({'role': 'assistant', 'content': response})
  ```

  This maintains **chat history and UI consistency**.

---

### 8️⃣ **Handling Errors**
- Errors are **caught and displayed** using Streamlit's `st.error()`.
  
  ```python
  except Exception as e:
      st.error(f"Error: {str(e)}")
  ```

  This prevents the chatbot from **crashing due to runtime issues**.

---

## 🚀 Running the Chatbot
To run the chatbot, execute the following command:
```sh
streamlit run filename.py
```
Make sure you have the required dependencies installed.

---

## 🛠️ Dependencies
Install required packages using:
```sh
pip install streamlit langchain langchain_community langchain_huggingface faiss-cpu huggingface_hub python-dotenv
```

---

## 🔗 Summary
This chatbot follows a **retrieval-augmented generation (RAG) approach**, integrating:
1. **Vector retrieval (FAISS + HuggingFaceEmbeddings)** 📌
2. **Language model inference (ChatGroq - Gemma2-9b-It)** 🤖
3. **Streamlit-based UI** 🎨

This makes it **efficient, context-aware, and easy to interact with**.

---

## 📜 Future Improvements
- ✅ **Improve retrieval quality** with better embeddings.
- ✅ **Enhance UI** with better chat history visualization.
- ✅ **Experiment with other LLMs** like `GPT-4`, `Claude`, or `LLaMA`.

---

## 📧 Contact
For any questions or improvements, feel free to reach out!


![alt text](<Screenshot 2025-03-27 122318.png>)
