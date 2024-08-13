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

from typing import List,Tuple
#import gradio as gr
from llama_index.core import Settings

# openai
import tiktoken


# open-source
from transformers import AutoTokenizer

from llama_index.core import ChatPromptTemplate
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
# Text QA Prompt
chat_text_qa_msgs = [
    (
        "system",
        "Always answer the question, even if the context isn't helpful.",
    ),
    ("user", qa_prompt_str),
]
text_qa_template = ChatPromptTemplate.from_messages(chat_text_qa_msgs)

# Refine Prompt
chat_refine_msgs = [
    (
        "system",
        "Always answer the question, even if the context isn't helpful.",
    ),
    ("user", refine_prompt_str),
]
refine_template = ChatPromptTemplate.from_messages(chat_refine_msgs)


import shutil
import os


def random_response(message, history):
    return random.choice(["Yes", "No"])

def upload_file(file):
    gradio.Info("Your file is being read!!!...You will be notified once it is done!!!")
    UPLOAD_FOLDER = "./data13"
    if not os.path.exists(UPLOAD_FOLDER):
        os.mkdir(UPLOAD_FOLDER)
    shutil.copy(file, UPLOAD_FOLDER)
    documents = SimpleDirectoryReader(input_files=[file]).load_data()
    llm = Ollama(model="llama3", request_timeout=2000.0)
    
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Settings.llm = llm
    Settings.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo").encode
    #Settings.tokenzier = AutoTokenizer.from_pretrained(
    #    "mistralai/Mixtral-8x7B-Instruct-v0.1"
    #)
    index = VectorStoreIndex.from_documents(documents)
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("quickstart")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context)
    gradio.Info("Your file is read!!!... Now  post your query!!!")

def doc_genie(message, history):
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = Ollama(model="llama3", request_timeout=2000.0)
    


    Settings.llm = llm
    Settings.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo").encode
    #Settings.tokenzier = AutoTokenizer.from_pretrained(
        #"mistralai/Mixtral-8x7B-Instruct-v0.1"
    #)
    #documents = SimpleDirectoryReader(input_files=["./data13/infosys-ar-23.pdf"]).load_data()
    #index2 = VectorStoreIndex.from_documents(documents)
    #query_engine = index2.as_query_engine(streaming=True,llm=llm)
    
    db2 = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db2.get_or_create_collection("quickstart")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index2 = VectorStoreIndex.from_vector_store(vector_store)
    query_engine = index2.as_query_engine(streaming=True,llm=llm)
    #chat_engine = index.as_chat_engine(chat_mode="condense_question", llm=llm, verbose=True, text_qa_template=text_qa_template,refine_template=refine_template)
    #if message == "Hi":
        #response=llm.complete(message)
    #    output=str(llm.complete(message))
    #    print(output)
    #    words = output.split()
    #    ans = ""
    #    for word in words:
    #        ans = ans +" "+ word
    #        yield ans

        #print(response)
        #print(type(response))
    #    ans = ""
        #for token in response:
        #    ans = ans + token
        #yield response
    #else:
        #response = chat_engine.stream_chat(message)
    response = query_engine.query(message)

    #print(response.chat_stream)
    ans=""
    for token in response.response_gen:
        ans = ans  + token
        yield ans



def genie(message, history):
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = Ollama(model="llama3", request_timeout=2000.0)
    #llm = Ollama(model="llama2", request_timeout=2000.0)


    Settings.llm = llm
    #Settings.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo").encode
    #Settings.tokenzier = AutoTokenizer.from_pretrained(
    #    "mistralai/Mixtral-8x7B-Instruct-v0.1"
    #)
    
    response = str(llm.complete(message))
    #response = str(llm)
    #print(response)
    words = response.split()    #print(response.chat_stream)
    ans=""
    for token in words:
        #print(token)
        ans = ans+" "+token
        #print(ans)
        time.sleep(0.1)

        yield ans
    
    #response1 = llm.stream_complete(message)
    #rint(response1)
    #for token in str(llm.stream_chat(message)):
    #    ans = ans+" "+token
    #    yield ans

    #output=response_stream.print_response_stream()
    #output= response_stream.response_gen
    """
    for i in range(len(response_stream.response_gen)):
        time.sleep(0.05)
        
        yield i + output
    """
    #response_stream.print_response_stream()
    #print(response_stream.response_gen)
    #print(response_stream.get_response())
    #for i in response_stream.print_response_stream():
    #    yield i

    #ans = ""
    #output = response_stream.get_response()
    #print(type(output))
    #for token in output:
    #    ans = ans+" "+token
    #    yield ans

    #yield
    #return output.

    #return output

#demo.launch()





private_GPT = gradio.ChatInterface(genie,fill_height=True).queue()
custom_GPT = gradio.ChatInterface(doc_genie,fill_height=True).queue()

with gradio.Blocks() as demo:
    gradio.Markdown(
        """
        # Welcome!!!
        # Try the Conversational Agent, and RAG!!!
        """)
    #file_output = gr.File()
    upload_button = gradio.UploadButton("Click to Upload a File")
    upload_button.upload(upload_file, upload_button)
    gradio.TabbedInterface([private_GPT, custom_GPT], ["Agent", "Agent With RAG"]).queue()
    #gradio.ChatInterface(genie).queue()


if __name__ == "__main__":
    demo.launch()
