import random
import gradio
import time
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.llms.azure_openai import AzureOpenAI

from typing import List, Tuple
from llama_index.core import Settings

import tiktoken
from transformers import AutoTokenizer

from llama_index.core import ChatPromptTemplate
import shutil
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Prompt templates
qa_prompt_str = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the question: {query_str}\n"
)

refine_prompt_str = (
    "We have the opportunity to refine the original answer "
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "Given the new context, refine the original answer to better "
    "answer the question: {query_str}. "
    "If the context isn't useful, output the original answer again.\n"
    "Original Answer: {existing_answer}"
)

# Chat templates
chat_text_qa_msgs = [
    ("system", "Always answer the question, even if the context isn't helpful."),
    ("user", qa_prompt_str),
]
text_qa_template = ChatPromptTemplate.from_messages(chat_text_qa_msgs)

chat_refine_msgs = [
    ("system", "Always answer the question, even if the context isn't helpful."),
    ("user", refine_prompt_str),
]
refine_template = ChatPromptTemplate.from_messages(chat_refine_msgs)

def setup_llm_and_settings():
    llm = Ollama(model="mixtral:8x22b", request_timeout=100.0)
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Settings.llm = llm
    Settings.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo").encode
    return llm

def upload_file(file):
    logging.info("File upload started")
    UPLOAD_FOLDER = "./data13"
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    file_path = os.path.join(UPLOAD_FOLDER, file.name)
    shutil.copy(file.name, file_path)
    
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    llm = setup_llm_and_settings()
    
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("quickstart")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    
    logging.info("File processed and indexed successfully")
    return "Your file has been processed and indexed. You can now query the document."

def doc_genie(message, history):
    llm = setup_llm_and_settings()
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("quickstart")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store)
    query_engine = index.as_query_engine(streaming=True, llm=llm)
    
    response = query_engine.query(message)
    ans = ""
    for token in response.response_gen:
        ans += token
        yield ans

def genie(message, history):
    llm = setup_llm_and_settings()
    response = llm.complete(message)
    words = str(response).split()
    ans = ""
    for word in words:
        ans += f" {word}"
        time.sleep(0.05)
        yield ans.strip()

# Gradio interface
with gradio.Blocks() as demo:
    gradio.Markdown(
        """
        # Welcome to the Enhanced Conversational Agent!
        ## Try the Conversational Agent and RAG (Retrieval-Augmented Generation)!
        """
    )
    upload_button = gradio.UploadButton("Click to Upload a File")
    file_status = gradio.Textbox(label="File Status")
    upload_button.upload(upload_file, upload_button, file_status)
    
    private_GPT = gradio.ChatInterface(genie, title="General Conversation", description="Chat with the AI without document context")
    custom_GPT = gradio.ChatInterface(doc_genie, title="Document-based Chat", description="Chat with the AI using the uploaded document as context")
    
    gradio.TabbedInterface([private_GPT, custom_GPT], ["General Agent", "Document-Aware Agent"])

if __name__ == "__main__":
    demo.launch()
