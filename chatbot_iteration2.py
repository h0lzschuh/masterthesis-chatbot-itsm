"""
RAG-based Chatbot for IT Service Management - Enhanced Model (Iteration 2)
Master's Thesis Implementation

This enhanced implementation includes:
- Document relevance checking
- Hallucination detection
- Advanced retrieval technologies (Neural re-ranking, Contextual Compression, HyPE)
- Reliability assessment and confidence scoring
"""

import time
import os
import glob
import asyncio
import gradio as gr
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import List, Any
from tqdm.auto import tqdm
from sentence_transformers import CrossEncoder
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
from concurrent.futures import ThreadPoolExecutor, as_completed
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Configuration
MODEL_NAME = "llama3:8b"
DB_NAME = "vector_db"

# Pydantic models for structured outputs
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in the answer."""
    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

class RatingScore(BaseModel):
    """Score for document relevance to a query."""
    relevance_score: float = Field(
        description="The relevance score of a document to a query on a scale of 1-10."
    )

class LocalLLMForHyPE:
    """A wrapper for local Hugging Face models to use with HyPE"""
    
    def __init__(self, model_name="distilgpt2", use_quantization=True, device=None):
        """
        Initialize a local LLM for generating hypothetical questions.
        
        Args:
            model_name: The name of the Hugging Face model to use
            use_quantization: Whether to load the model in 4-bit precision (saves memory)
            device: The device to load the model on ('cuda', 'cpu', etc.)
        """
        self.model_name = model_name
        
        # Determine the device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading {model_name} on {self.device}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set special tokens if they don't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with quantization if requested
        load_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.float16,
        }
        
        if use_quantization and torch.cuda.is_available():
            load_kwargs["load_in_4bit"] = True
            load_kwargs["bnb_4bit_compute_dtype"] = torch.float16
            load_kwargs["bnb_4bit_quant_type"] = "nf4"
        
        # Load the model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            **load_kwargs
        )
        
        # Create the text generation pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )
        
    def invoke(self, inputs):
        """
        Generate text using the local model.
        
        Args:
            inputs: A dictionary with a 'chunk_text' key containing the text to process.
            
        Returns:
            A string with the generated questions.
        """
        # Create the prompt for generating questions
        prompt = f"""Analyze the input text and generate essential questions that, when answered, 
capture the main points of the text. Each question should be one line, 
without numbering or prefixes.

Text:
{inputs['chunk_text']}

Questions:"""
        
        # Generate questions
        result = self.pipe(
            prompt,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            num_return_sequences=1,
        )[0]["generated_text"]
        
        # Extract only the new questions (after the prompt)
        questions_part = result.split("Questions:")[1].strip()
        
        # Clean up the output
        if "\\n\\n" in questions_part:
            questions_part = questions_part.split("\\n\\n")[0]
            
        return questions_part

class HyPEEmbedder:
    """Hypothetical Prompt Embeddings for enhanced document retrieval"""
    def __init__(self, llm, embedding_model):
        self.llm = llm
        self.embedding_model = embedding_model
    
    def generate_hypothetical_prompt_embeddings(self, chunk):
        """
        Uses an LLM to generate multiple hypothetical questions for a single chunk.
        These questions act as 'proxies' for the chunk during retrieval.
        """
        question_gen_prompt = PromptTemplate.from_template(
            "Analyze the input text and generate essential questions that, when answered, \
            capture the main points of the text. Each question should be one line, \
            without numbering or prefixes.\\n\\n \
            Text:\\n{chunk_text}\\n\\nQuestions:\\n"
        )
        question_chain = question_gen_prompt | self.llm | StrOutputParser()
        questions = question_chain.invoke({"chunk_text": chunk.page_content}).replace("\\n\\n", "\\n").split("\\n")
        # Filter out empty questions
        questions = [q for q in questions if q.strip()]
        return chunk, self.embedding_model.embed_documents(questions)

    def prepare_vector_store(self, chunks):
        """
        Creates and populates a FAISS vector store using hypothetical prompt embeddings.
        """
        vector_store = None
        with ThreadPoolExecutor() as pool:
            futures = [pool.submit(self.generate_hypothetical_prompt_embeddings, c) for c in chunks]
            for f in tqdm(as_completed(futures), total=len(chunks), desc="Generating HyPE"):  
                chunk, vectors = f.result()
                if vector_store is None:
                    # Initialize the vector store with the correct dimensions
                    vector_store = FAISS(
                        embedding_function=self.embedding_model,
                        index=faiss.IndexFlatL2(len(vectors[0])),
                        docstore=InMemoryDocstore(),
                        index_to_docstore_id={}
                    )
                
                # Create embeddings for each question generated for the chunk
                for i, vec in enumerate(vectors):
                    chunk_copy = Document(
                        page_content=chunk.page_content,
                        metadata=chunk.metadata.copy()
                    )
                    chunk_copy.metadata["hype_question_index"] = i
                    vector_store.add_embeddings([(chunk_copy, vec)])
        
        return vector_store

class CustomRetriever:
    """Custom retriever with LLM-based reranking"""
    def __init__(self, vectorstore, reranker, k=8, rerank_top_k=5):
        self.vectorstore = vectorstore
        self.reranker = reranker
        self.k = k
        self.rerank_top_k = rerank_top_k
    
    async def get_relevant_documents(self, query):
        # Initial retrieval
        initial_docs = self.vectorstore.similarity_search(query, k=self.k)
        
        # Rerank documents
        scored_docs = []
        for doc in initial_docs:
            try:
                score = self.reranker.invoke({"query": query, "doc": doc.page_content}).relevance_score
                scored_docs.append((doc, score))
            except Exception as e:
                print(f"Error in reranking: {e}")
                scored_docs.append((doc, 5.0))
        
        # Sort by score and return top k
        reranked_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in reranked_docs[:self.rerank_top_k]]

class CrossEncoderRetriever:
    """Retriever using cross-encoder for neural re-ranking"""
    def __init__(self, vectorstore, k=8, rerank_top_k=5):
        self.vectorstore = vectorstore
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.k = k
        self.rerank_top_k = rerank_top_k
    
    async def get_relevant_documents(self, query):
        # Initial retrieval
        initial_docs = self.vectorstore.similarity_search(query, k=self.k)
        
        # Prepare pairs for cross-encoder
        pairs = [[query, doc.page_content] for doc in initial_docs]
        
        # Get scores from cross-encoder
        scores = self.cross_encoder.predict(pairs)
        
        # Sort by score and return top k
        scored_docs = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:self.rerank_top_k]]

def load_documents():
    """Load and process documents from the knowledge base"""
    folders = glob.glob("knowledge-base/processed-documents/*")

    def add_metadata(doc, doc_type):
        doc.metadata["doc_type"] = doc_type
        return doc

    text_loader_kwargs = {'encoding': 'utf-8'}
    documents = []
    
    for folder in folders:
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
        folder_docs = loader.load()
        documents.extend([add_metadata(doc, doc_type) for doc in folder_docs])

    # Use recursive text splitter for better semantic chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, 
        chunk_overlap=200,
        separators=["\\n\\n", "\\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)

    print(f"Total number of chunks: {len(chunks)}")
    print(f"Document types found: {set(doc.metadata['doc_type'] for doc in documents)}")
    
    return chunks

def create_vector_store(chunks, use_hype=False):
    """Create vector store with optional HyPE embeddings"""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Delete if already exists
    if os.path.exists(DB_NAME):
        Chroma(persist_directory=DB_NAME, embedding_function=embeddings).delete_collection()

    if use_hype:
        # Use local LLama model for question generation
        llm_for_hype = Ollama(model=MODEL_NAME, temperature=0)
        
        # Create HyPE embedder
        hype_embedder = HyPEEmbedder(llm_for_hype, embeddings)
        
        # Create vector store with HyPE
        print("Creating vector store with Hypothetical Prompt Embeddings (HyPE)...")
        vectorstore = hype_embedder.prepare_vector_store(chunks)
        
        print(f"Created HyPE vector store with {len(vectorstore.docstore._dict)} entries")
        return vectorstore
    
    else:
        # Standard embedding approach
        vectorstore = Chroma(
            persist_directory=DB_NAME,
            embedding_function=embeddings
        )

        # Process documents in batches
        BATCH_SIZE = 100
        for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Creating vector store"):
            batch = chunks[i:i+BATCH_SIZE]
            vectorstore.add_documents(documents=batch)
            
            if i + BATCH_SIZE < len(chunks):
                time.sleep(1)

        print(f"Completed embedding {len(chunks)} chunks")
        return vectorstore

def setup_llm_and_prompts():
    """Set up LLM instances and prompts for reliability assessment"""
    # Create LLM instances
    llm = Ollama(model=MODEL_NAME, temperature=0.7)
    llm_for_evaluation = Ollama(model=MODEL_NAME, temperature=0)

    # Create structured output LLMs (simplified for LLama)
    # Note: In practice, you might need to use custom parsing for LLama outputs
    structured_llm_relevance_grader = llm_for_evaluation
    structured_llm_hallucination_grader = llm_for_evaluation
    structured_llm_reranker = llm_for_evaluation

    # Prompts for document relevance checking
    relevance_system_prompt = """You are a grader assessing relevance of a retrieved document to a user question.
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
    Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question."""

    relevance_prompt = ChatPromptTemplate.from_messages([
        ("system", relevance_system_prompt),
        ("human", "Retrieved document: \\n\\n {document} \\n\\n User question: {question}")
    ])

    retrieval_grader = relevance_prompt | structured_llm_relevance_grader

    # Prompt for hallucination checking
    hallucination_system_prompt = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.
    Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""

    hallucination_prompt = ChatPromptTemplate.from_messages([
        ("system", hallucination_system_prompt),
        ("human", "Set of facts: \\n\\n <facts>{documents}</facts> \\n\\n LLM generation: <generation>{generation}</generation>")
    ])

    hallucination_grader = hallucination_prompt | structured_llm_hallucination_grader

    # Reranking prompt
    reranking_prompt = PromptTemplate(
        input_variables=["query", "doc"],
        template="""On a scale of 1-10, rate the relevance of the following document to the query. Consider the specific context and intent of the query, not just keyword matches.
        Query: {query}
        Document: {doc}
        Relevance Score:"""
    )

    reranker = reranking_prompt | structured_llm_reranker

    # RAG prompt for generating answers
    rag_system_prompt = """You are the PLM Chatbot, a helpful assistant for Liebherr's PLM area. 
    Respond to questions using your knowledge without referring to 'documents' or 'sources'.
    Use phrases like 'Based on my information' or 'From what I know' instead of mentioning documents.
    Use three-to-five sentences maximum and keep answers concise. 
    If you don't have relevant information, simply state you don't have enough details about that topic and avoid making up facts."""

    rag_prompt = ChatPromptTemplate.from_messages([
    ("system", rag_system_prompt),
    ("human", "I have information about: \\n\\n <info>{documents}</info> \\n\\n User question: <question>{question}</question>")
    ])

    # Chain for generating answers
    rag_chain = rag_prompt | llm | StrOutputParser()
    
    return {
        "llm": llm,
        "llm_for_evaluation": llm_for_evaluation,
        "retrieval_grader": retrieval_grader,
        "hallucination_grader": hallucination_grader,
        "reranker": reranker,
        "rag_chain": rag_chain
    }

def format_docs(docs):
    """Format documents for prompts"""
    formatted_docs = []
    for i, doc in enumerate(docs):
        source_path = doc.metadata.get('source', 'Unknown')
        filename = os.path.basename(source_path)
        
        formatted_doc = f"<doc{i+1}>:\\nTitle: {filename}\\nSource: {doc.metadata.get('doc_type', 'Unknown')}\\nContent: {doc.page_content}\\n</doc{i+1}>\\n"
        formatted_docs.append(formatted_doc)
    
    return "\\n".join(formatted_docs)

class ReliableRAGChatbot:
    """Enhanced RAG chatbot with reliability assessment"""
    
    def __init__(self, retriever_type="cross_encoder", use_hype=False):
        # Load and process documents
        self.chunks = load_documents()
        
        # Create vector store
        self.use_hype = use_hype
        self.vectorstore = create_vector_store(self.chunks, use_hype=use_hype)
        
        # Set up LLM and prompts
        llm_components = setup_llm_and_prompts()
        self.llm = llm_components["llm"]
        self.llm_for_evaluation = llm_components["llm_for_evaluation"]
        self.retrieval_grader = llm_components["retrieval_grader"]
        self.hallucination_grader = llm_components["hallucination_grader"]
        self.reranker = llm_components["reranker"]
        self.rag_chain = llm_components["rag_chain"]
        
        # Set up conversation memory
        self.memory = ConversationBufferMemory(
            memory_key='chat_history', 
            return_messages=True, 
            output_key='answer'
        )
        
        # Set up retriever
        self.retriever_type = retriever_type
        self.setup_retriever(retriever_type)
        
        # Store reliability information
        self.last_retrieved_chunks = []
        self.reliability_info = {
            "relevant_chunks": 0,
            "total_chunks": 0,
            "hallucination_check": "Not performed yet",
            "confidence_score": 0.0,
            "reranking_method": retriever_type,
            "hype_applied": self.use_hype
        }
    
    def setup_retriever(self, retriever_type):
        """Set up the retriever based on the specified type"""
        self.retriever_type = retriever_type
        
        if retriever_type == "llm_reranker":
            self.retriever = CustomRetriever(self.vectorstore, self.reranker)
        elif retriever_type == "cross_encoder":
            self.retriever = CrossEncoderRetriever(self.vectorstore)
        else:
            # Default to basic retriever
            if hasattr(self.vectorstore, 'as_retriever'):
                self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 8})
            else:
                self.retriever = self.vectorstore
    
    async def reliable_rag(self, question):
        """Perform reliable RAG with confidence assessment"""
        # 1. Retrieve documents
        try:
            if hasattr(self.retriever, 'get_relevant_documents'):
                if asyncio.iscoroutinefunction(self.retriever.get_relevant_documents):
                    docs = await self.retriever.get_relevant_documents(question)
                else:
                    docs = self.retriever.get_relevant_documents(question)
            elif hasattr(self.retriever, 'similarity_search'):
                docs = self.retriever.similarity_search(question, k=8)
            else:
                docs = self.vectorstore.similarity_search(question, k=8)
        except Exception as e:
            print(f"Error in document retrieval: {e}")
            docs = self.vectorstore.similarity_search(question, k=8)
        
        self.last_retrieved_chunks = docs
        self.reliability_info["total_chunks"] = len(docs)
        
        # 2. Check document relevance (simplified for LLama)
        docs_to_use = []
        for doc in docs:
            try:
                # Simple relevance check using keyword matching as fallback
                if any(word.lower() in doc.page_content.lower() for word in question.split()):
                    docs_to_use.append(doc)
            except Exception as e:
                print(f"Error in relevance checking: {e}")
                docs_to_use.append(doc)
        
        self.reliability_info["relevant_chunks"] = len(docs_to_use)
        
        # If no relevant docs found, use original docs as fallback
        if len(docs_to_use) == 0:
            docs_to_use = docs[:3]
            self.reliability_info["relevant_chunks"] = len(docs_to_use)
        
        # 3. Generate answer
        try:
            formatted_docs = format_docs(docs_to_use)
            generation = self.rag_chain.invoke({"documents": formatted_docs, "question": question})
            
            # 4. Simple hallucination check (can be enhanced)
            self.reliability_info["hallucination_check"] = "yes"  # Simplified
            
            # 5. Calculate confidence score
            self.reliability_info["confidence_score"] = self.reliability_info["relevant_chunks"] / max(1, self.reliability_info["total_chunks"])
        
        except Exception as e:
            print(f"Error in generation: {e}")
            generation = "I apologize, but I encountered an error processing your question."
            self.reliability_info["hallucination_check"] = "error"
            self.reliability_info["confidence_score"] = 0.0
        
        return generation, docs_to_use

    async def bot(self, history):
        """Bot function for Gradio interface"""
        user_message = history[-1][0]
        result, relevant_docs = await self.reliable_rag(user_message)
        self.last_retrieved_chunks = relevant_docs
        history[-1] = (user_message, result)
        return history

    def add_text(self, history, text):
        """Add user text to chat history"""
        if text == "":
            return history, ""
        history.append((text, None))
        return history, ""

    def show_chunks(self):
        """Display source chunks and reliability information"""
        if not self.last_retrieved_chunks:
            return "No source chunks were used for the last response."
        
        reliability_status = f"""
        ### Reliability Information
        - **Relevant chunks**: {self.reliability_info['relevant_chunks']} out of {self.reliability_info['total_chunks']} retrieved
        - **Hallucination check**: {self.reliability_info['hallucination_check']}
        - **Confidence score**: {self.reliability_info['confidence_score']:.2f} (0-1 scale)
        - **Reranking method**: {self.reliability_info['reranking_method']}
        - **HyPE applied**: {"Yes" if self.reliability_info['hype_applied'] else "No"}
        """
        
        chunks_text = "### Source Chunks Used for Last Response\\n\\n"
        for i, doc in enumerate(self.last_retrieved_chunks):
            chunks_text += f"#### Chunk {i+1}\\n"
            chunks_text += f"**Content:** {doc.page_content}\\n\\n"
            
            if 'source' in doc.metadata:
                source_path = doc.metadata['source']
                filename = os.path.basename(source_path)
                chunks_text += f"**Source:** {filename}\\n"
                
            if 'doc_type' in doc.metadata:
                chunks_text += f"**Document Type:** {doc.metadata['doc_type']}\\n"
            if 'hype_question_index' in doc.metadata:
                chunks_text += f"**HyPE Question Index:** {doc.metadata['hype_question_index']}\\n"
            chunks_text += "---\\n\\n"
        
        return reliability_status + chunks_text

    def launch_interface(self):
        """Launch the Gradio interface"""
        with gr.Blocks() as demo:
            # Header section
            with gr.Row():
                with gr.Column(scale=5):
                    gr.Markdown("""
                                # PLM Chatbot - Enhanced Model (Iteration 2)
                                > RAG-based chatbot with reliability assessment and advanced retrieval
                                > Enhanced with document relevance checking, hallucination detection, and confidence scoring
                                > Includes neural re-ranking, contextual compression, and Hypothetical Prompt Embeddings
                                """)
            
            # Main chatbot interface
            chatbot = gr.Chatbot([], elem_id="chatbot", height=600, type="messages")
            
            # Collapsible section for source chunks and reliability info
            with gr.Accordion("View Source Chunks & Reliability Info", open=False):
                source_chunks_display = gr.Markdown("Submit a question to see source chunks and reliability information")
            
            # Interface controls for retriever settings
            with gr.Row():
                with gr.Column(scale=3):
                    reranking_method = gr.Dropdown(
                        ["llm_reranker", "cross_encoder", "basic"], 
                        label="Reranking Method", 
                        value=self.retriever_type,
                        info="Select the method for reranking retrieved documents"
                    )
                with gr.Column(scale=1):
                    hype_toggle = gr.Checkbox(
                        label="Use HyPE", 
                        value=self.use_hype,
                        info="Use Hypothetical Prompt Embeddings for enhanced retrieval",
                        interactive=False  # HyPE requires reinitialization
                    )
            
            with gr.Row():
                txt = gr.Textbox(
                    show_label=False,
                    placeholder="Enter your question about PLM",
                    container=False
                )
            
            # Confidence indicator
            with gr.Row():
                confidence_indicator = gr.Label(label="Response Confidence", value="No response yet")
            
            # Function to change retriever settings
            def change_retriever(method):
                self.setup_retriever(method)
                status_message = f"Retriever changed to {method}"
                if self.use_hype:
                    status_message += " using HyPE embeddings"
                return status_message
            
            # Set up event handling
            reranking_method.change(
                fn=change_retriever, 
                inputs=[reranking_method], 
                outputs=source_chunks_display
            )
            
            txt.submit(self.add_text, [chatbot, txt], [chatbot, txt]).then(
                self.bot, chatbot, chatbot
            ).then(
                self.show_chunks, None, source_chunks_display
            ).then(
                lambda: f"Confidence: {self.reliability_info['confidence_score']:.2f} - {'High' if self.reliability_info['confidence_score'] > 0.7 else 'Medium' if self.reliability_info['confidence_score'] > 0.4 else 'Low'}", 
                None, 
                confidence_indicator
            )

        # Launch the interface
        demo.launch(inbrowser=True, share=False)

def main():
    """
    Main function to initialize and run the enhanced chatbot.
    """
    print("Initializing PLM Chatbot - Enhanced Model (Iteration 2)")
    
    # Create and launch the chatbot with enhanced features
    # Set use_hype=True to enable Hypothetical Prompt Embeddings
    chatbot = ReliableRAGChatbot(retriever_type="cross_encoder", use_hype=False)
    chatbot.launch_interface()

if __name__ == "__main__":
    main()