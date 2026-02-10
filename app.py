import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# use the correct name of the specific class you need
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings # <-- Correct Import
from langchain_classic.chains  import RetrievalQA
from langchain_groq import ChatGroq

# --- Configuration Constants ---
DATA_PATH = "docs"
CHROMA_PATH = "chroma_data"

# --- Data Engineering / ETL Functions ---

def load_documents():
    """Loads all PDF documents from the DATA_PATH folder."""
    if not os.path.exists(DATA_PATH) or not os.listdir(DATA_PATH):
        st.error(f"Error: The '{DATA_PATH}' folder is empty or doesn't exist.")
        st.stop()
        
    all_files = [os.path.join(DATA_PATH, f) for f in os.listdir(DATA_PATH) if f.endswith(".pdf")]
    
    loaders = [PyPDFLoader(f) for f in all_files]
    documents = []
    for loader in loaders:
        documents.extend(loader.load())
    return documents

def split_text(documents: list):
    """Splits documents into smaller, manageable chunks for the Vector Store."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_documents(documents)

def create_vector_store(chunks):
    """Creates embeddings and stores them in a Vector Database (ChromaDB)."""
    
    # Initialize the stable, local Sentence Transformer model
    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2" # Runs locally via sentence-transformers library
    )
    
    vector_store = Chroma.from_documents(
        chunks, 
        embeddings, 
        persist_directory=CHROMA_PATH
    )
    vector_store.persist()
    return vector_store

def prepare_knowledge_base():
    """Runs the ETL/RAG pre-processing pipeline once or loads the existing store."""
    try:
        if not os.path.exists(CHROMA_PATH):
            # --- Pipeline execution (Creation) ---
            st.info("Knowledge base not found. Running ETL pipeline...")
            
            documents = load_documents()
            st.info(f"Splitting {len(documents)} pages into chunks...")
            chunks = split_text(documents)
            
            st.info(f"Creating vector store with {len(chunks)} chunks. This may take a moment...")
            vector_store = create_vector_store(chunks)
            st.success("Knowledge Base is Ready!")
            return vector_store
        else:
            # --- Load existing database (Loading) ---
            st.info("Loading existing knowledge base from disk...")
            
            # Use the correct class definition for loading
            embeddings = SentenceTransformerEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )
            
            vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
            st.success("Knowledge Base Loaded!")
            return vector_store
    except Exception as e:
        # This will catch errors during the loading/creation process
        st.error(f"An error occurred during data preparation: {e}")
        st.stop()


# --- RAG Agent Logic ---

def setup_rag_chain(vector_store):
    """Sets up the Retrieval-Augmented Generation (RAG) chain using Groq."""
    
    # 1. Initialize the LLM using Groq
    llm = ChatGroq(
        temperature=0, 
        groq_api_key=st.secrets["groq_api_key"],
        # *** FIX: Use the recommended replacement model ***
        model_name="llama-3.1-8b-instant" 
    )
    
    # 2. Create the Retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 4}) 
    
    # 3. Create the RAG Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True 
    )
    return qa_chain


# --- Streamlit Application ---

def main():
    st.set_page_config(page_title="Credit Risk Regulation Q&A Agent", layout="wide")
    st.title("🏛️ Credit Risk Regulation Q&A Agent (Powered by Groq)")
    st.markdown(
        """
        This RAG agent uses **Llama 3.1** to provide accurate, grounded answers based *only* on the 
        documents provided, showcasing a production-ready data pipeline for regulatory compliance.
        """
    )
    
    if "groq_api_key" not in st.secrets:
        st.error("GROQ API Key not found in .streamlit/secrets.toml. Please add it to proceed.")
        st.stop()
    
    # Run the Data Prep pipeline
    with st.spinner("Preparing Knowledge Base..."):
        vector_store = prepare_knowledge_base()
        qa_chain = setup_rag_chain(vector_store)

    # User input form
    with st.form("rag_form", clear_on_submit=True):
        user_query = st.text_area(
            "Enter your question (e.g., 'What are the three pillars of Basel II and how are they defined?')",
            key="user_query_input"
        )
        submitted = st.form_submit_button("Get Answer", type="primary")

    if submitted and user_query:
        # Run the RAG query
        with st.spinner("Searching knowledge base and generating response..."):
            response = qa_chain.invoke({"query": user_query})
            
            # Display the result
            st.subheader("✅ AI Answer")
            st.info(response["result"])
            
            # Display the sources for verification
            st.subheader("🔎 Source Documents (Evidence)")
            if response["source_documents"]:
                for i, doc in enumerate(response["source_documents"]):
                    st.markdown(
                        f"""
                        **Source {i+1}** (File: `{os.path.basename(doc.metadata.get('source'))}` | Page: `{doc.metadata.get('page')}`)
                        > *...{doc.page_content.strip().replace('\\n', ' ')}...*
                        """
                    )
            else:
                st.warning("No relevant source documents were retrieved for this query.")

if __name__ == "__main__":
    main()