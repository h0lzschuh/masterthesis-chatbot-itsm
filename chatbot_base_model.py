"""
RAG-based Chatbot for IT Service Management - Base Model
Master's Thesis Implementation

This implementation uses a simple RAG (Retrieval Augmented Generation) approach
with LLama 3 8B for question answering in the PLM domain.
"""

import time
import os
import glob
import gradio as gr
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

# Configuration
MODEL_NAME = "llama3:8b"
DB_NAME = "vector_db"

def load_and_process_documents(knowledge_base_path):
    """
    Load documents from the knowledge base and split them into chunks.
    
    Args:
        knowledge_base_path (str): Path to the knowledge base directory
        
    Returns:
        list: List of document chunks
    """
    folders = glob.glob(f"{knowledge_base_path}/*")
    
    def add_metadata(doc, doc_type):
        doc.metadata["doc_type"] = doc_type
        return doc
    
    text_loader_kwargs = {'encoding': 'utf-8'}
    documents = []
    
    for folder in folders:
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(
            folder, 
            glob="**/*.md", 
            loader_cls=TextLoader, 
            loader_kwargs=text_loader_kwargs
        )
        folder_docs = loader.load()
        documents.extend([add_metadata(doc, doc_type) for doc in folder_docs])
    
    # Split documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    print(f"Total number of chunks: {len(chunks)}")
    print(f"Document types found: {set(doc.metadata['doc_type'] for doc in documents)}")
    
    return chunks

def create_vector_store(chunks):
    """
    Create a vector store from document chunks using HuggingFace embeddings.
    
    Args:
        chunks (list): List of document chunks
        
    Returns:
        Chroma: Vector store instance
    """
    # Use local HuggingFace embeddings for privacy
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Delete existing vector store if it exists
    if os.path.exists(DB_NAME):
        existing_store = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
        existing_store.delete_collection()
    
    # Create new vector store
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=DB_NAME
    )
    
    print(f"Vectorstore created with {vectorstore._collection.count()} documents")
    return vectorstore

def setup_conversation_chain(vectorstore):
    """
    Set up the conversational retrieval chain with LLama 3 8B.
    
    Args:
        vectorstore: Vector store for document retrieval
        
    Returns:
        ConversationalRetrievalChain: Configured conversation chain
    """
    # Initialize LLama 3 8B model via Ollama
    llm = Ollama(model=MODEL_NAME, temperature=0.7)
    
    # Set up conversation memory
    memory = ConversationBufferMemory(
        memory_key='chat_history', 
        return_messages=True
    )
    
    # Configure retriever to get top 25 chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 25})
    
    # Create conversational retrieval chain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=retriever, 
        memory=memory
    )
    
    return conversation_chain

def create_chatbot_interface(conversation_chain):
    """
    Create and launch the Gradio interface for the chatbot.
    
    Args:
        conversation_chain: Configured conversation chain
    """
    # System prompt for the PLM chatbot
    system_prompt = """
    Please keep in mind:
    1. You are a polite chatbot that helps in the company's area of PLM. Do not speak about other areas.
    2. Always call yourself PLM Chatbot, never refer to the LLM used.
    3. Never use any names in the responses.
    4. If a request sounds like something new in the PLM area shall be developed, always mention to include working group members for the specification.
    5. Structure your answers clearly.
    """
    
    def chat_function(question, history):
        """
        Process user questions and return chatbot responses.
        
        Args:
            question (str): User's question
            history: Chat history (managed by Gradio)
            
        Returns:
            str: Chatbot response
        """
        result = conversation_chain.invoke({"question": system_prompt + question})
        return result["answer"]
    
    # Create and launch Gradio interface
    interface = gr.ChatInterface(
        chat_function,
        type="messages",
        title="PLM Chatbot - Base Model",
        description="RAG-based chatbot for IT Service Management in the PLM domain"
    )
    
    interface.launch(inbrowser=True, share=False)

def main():
    """
    Main function to initialize and run the chatbot.
    """
    print("Initializing PLM Chatbot - Base Model")
    print("Loading and processing documents...")
    
    # Load documents (adjust path as needed)
    knowledge_base_path = "knowledge-base/processed-documents"
    chunks = load_and_process_documents(knowledge_base_path)
    
    print("Creating vector store...")
    vectorstore = create_vector_store(chunks)
    
    print("Setting up conversation chain...")
    conversation_chain = setup_conversation_chain(vectorstore)
    
    print("Launching chatbot interface...")
    create_chatbot_interface(conversation_chain)

if __name__ == "__main__":
    main()
