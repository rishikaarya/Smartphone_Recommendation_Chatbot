import streamlit as st
import pandas as pd
import chromadb
from llama_index.core import StorageContext, VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.readers.file import PagedCSVReader
from llama_index.core.readers import SimpleDirectoryReader, Document

# 1️⃣ **Define Embedding Model**
embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://localhost:11434",
)

# 2️⃣ **Configure LlamaIndex Global Settings**
Settings.embed_model = embed_model
Settings.llm = Ollama(
    model="qwen2.5:latest",
    request_timeout=120.0,
    temperature=0.0,
    mirostat=0
)

# 3️⃣ **Load CSV Data**
file_path = "cpai_final.csv"
df = pd.read_csv(file_path)

# 4️⃣ **Prepare Documents for Indexing**
columns_of_interest = ['Product Title', 'Price', 'battery', 'camera', 'display', 
                        'performance', 'price', 'design', 'charging', 'sound', 'Cumulative_Score']

documents = [Document(text=row.to_string()) for _, row in df[columns_of_interest].iterrows()]

# 5️⃣ **Setup ChromaDB Persistent Storage**
chroma_client = chromadb.PersistentClient(path="/home/ashok/Documents/chroma_db")

# 6️⃣ **Check and Create Collection**
collection_name = "datastore"
existing_collections = chroma_client.list_collections()

if collection_name not in existing_collections:
    chroma_collection = chroma_client.create_collection(collection_name)
else:
    chroma_collection = chroma_client.get_collection(collection_name)

# 7️⃣ **Create Chroma Vector Store**
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 8️⃣ **Index Documents**
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    show_progress=True
)

# 9️⃣ **Setup Query Engine**
query_engine = index.as_query_engine()

# 🔟 **Initialize Streamlit Session State for Chat History**
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Store past responses

# 🚀 **Streamlit Multi-Page UI**
st.set_page_config(layout="wide", page_title="Smartphone AI Assistant", page_icon="📱")

# Sidebar Navigation
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Chatbot", "About"])

# Sample Prompts
st.sidebar.header("💡 Example Prompts")
st.sidebar.write("""
- "Which smartphone has the best camera under ₹30,000?"
- "Suggest a phone with the best battery life."
- "Compare iPhone 15 and Samsung S23 Ultra."
""")

# AI Assistant Introduction
st.sidebar.header("🤖 Meet Your AI Guide")
st.sidebar.write("""
Hi! I'm your AI assistant, here to help you choose the best smartphone.
I analyze various specs like **camera, battery, performance, and design** to
suggest the best options based on your needs! 🔥
""")

# Home Page
if page == "Home":
    st.title("📱 Smart AI Assistant for Smartphone Recommendations")
    st.markdown(
        """
        #### Welcome to your AI-powered smartphone guide! 🎯
        - 🔍 Get instant **recommendations** based on your preferences.
        - 🏆 Compare models based on **battery, camera, display, and performance.**
        - 🤖 **Powered by AI** for accurate and up-to-date suggestions.
        """
    )
    
    with st.expander("📖 How to Use This Chatbot"):
        st.write("""
        **1️⃣ Enter your query** in the chat below (e.g., "Which phone has the best battery?").
        **2️⃣ AI will process and find the best match** based on stored smartphone data.
        **3️⃣ Get a list of recommendations** instantly!
        """)
    
    st.image("https://cdn.pixabay.com/photo/2017/01/06/19/15/smartphone-1957740_1280.jpg", use_column_width=True)

# Chatbot Page
elif page == "Chatbot":
    st.subheader("💬 Chat with the AI Assistant")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # User input
    user_query = st.chat_input("Ask about smartphone features...")
    
    if user_query:
        # Display user message
        st.chat_message("user").write(user_query)
        
        with st.spinner("Thinking..."):
            response = query_engine.query(user_query)  # Simulated AI response
        
        if hasattr(response, "source_nodes"):
            retrieved_docs = response.source_nodes  
            smartphone_names = []
            
            for doc in retrieved_docs:
                lines = doc.node.text.split("\n")
                for line in lines:
                    if "Product Title" in line:
                        smartphone_name = line.split(":", 1)[1].strip()
                        smartphone_names.append(smartphone_name)
            
            if smartphone_names:
                formatted_response = (
                    "Here are some smartphones that match your query: "
                    + ", ".join(smartphone_names) + "."
                )
            else:
                formatted_response = "I couldn't find any smartphones that match your request."
            
            # Display AI response
            with st.chat_message("assistant"):
                st.write(formatted_response)
            
            # Save response in history
            st.session_state.chat_history.append({"role": "assistant", "content": formatted_response})
        else:
            with st.chat_message("assistant"):
                st.write(response.response)
            
            # Save response in history
            st.session_state.chat_history.append({"role": "assistant", "content": response.response})

# About Page
elif page == "About":
    st.subheader("ℹ️ About This Chatbot")
    st.write("""
    This AI-powered chatbot helps users find the best smartphones based on their needs. 
    Using advanced AI techniques, it compares various models and suggests the best match. 
    The chatbot is trained on real-world smartphone data to ensure **accurate and up-to-date recommendations**.
    
    **Developed with ❤️ using LlamaIndex, ChromaDB & Streamlit**.
    """)