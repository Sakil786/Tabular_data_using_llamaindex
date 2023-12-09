![Llamaindex](https://github.com/Sakil786/Tabular_data_using_llamaindex/blob/main/img.png "Llamaindex")

### Tabular_data_using_llamaindex
**Imports and Environment Setup**
```python
import streamlit as st
from dotenv import load_dotenv
import torch
import sys
import os
from transformers import BitsAndBytesConfig

# llama_index
from langchain.embeddings import HuggingFaceInstructEmbeddings
from llama_index import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage, ServiceContext
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.indices.postprocessor import SimilarityPostprocessor, KeywordNodePostprocessor
from llama_index.response_synthesizers import get_response_synthesizer
from llama_index.text_splitter import SentenceSplitter
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.prompts import PromptTemplate
from llama_index.llms import HuggingFaceLLM

# chromadb
import chromadb
import logging
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
from llama_index import SummaryIndex
from llama_index.response.pprint_utils import pprint_response
from pathlib import Path
from llama_index import download_loader
from llama_index.response.pprint_utils import pprint_response
```

The code imports necessary libraries (streamlit, dotenv, torch, etc.) and several modules related to handling embeddings, databases, and querying.

**get_desired_llm() Function**
```python
def get_desired_llm():
  # Function to load LLM 
  hf_token="Hugging Face Token"
  quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
  llm = HuggingFaceLLM(
    model_name="meta-llama/Llama-2-7b-chat-hf",
    tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
    query_wrapper_prompt=PromptTemplate(" [INST] {query_str} [/INST] "),
    context_window=3900,
    model_kwargs={"token": hf_token, "quantization_config": quantization_config},
    tokenizer_kwargs={"token": hf_token},
    device_map="auto",
)
retrun llm
```
Defines a function to load and return a Language Model object.

**get_desired_embedding() Function**
```python
def get_desired_embedding():
#function to load Embedding
  embed_model = HuggingFaceInstructEmbeddings(
  model_name="hkunlp/instructor-large", model_kwargs={"device": DEVICE}
)
  return embed_model
```
Initializes and returns an Embedding

**get_csv_file() Function**
```python
def get_csv_file():
  #function to laod csv file
  SimpleCSVReader = download_loader("SimpleCSVReader")
  loader = SimpleCSVReader(encoding="utf-8")
  documents = loader.load_data(file=Path('/content/data/countries of the world.csv'))
  return documents
```
Loads data/documents from a CSV file.

**create_chunk() Function**
```python
def create_chunk():

  llm=get_desired_llm()
  embed_model=get_desired_embedding()
  documents=get_csv_file()
  text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20) # using the default chunk_# values as they work just fine

  # set context window
  context_window = 4096
  # set number of output tokens
  num_output = 256
  service_context = ServiceContext.from_defaults(llm=llm,
                                                embed_model=embed_model,
                                                text_splitter=text_splitter,
                                                #  context_window=context_window,
                                                #  num_output=num_output,
                                                )
  # # To make ephemeral client that is a short lasting client or an in-memory client
# db = chromadb.EphemeralClient()

  # initialize client, setting path to save data
  db = chromadb.PersistentClient(path="./chroma_db")

  # create collection
  chroma_collection = db.get_or_create_collection("csv_database")

  # assign chroma as the vector_store to the context
  vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
  storage_context = StorageContext.from_defaults(vector_store=vector_store)

  # create your index
  vector_index = VectorStoreIndex.from_documents(
      documents,
      storage_context=storage_context,
      service_context=service_context
  )
  return vector_index
```
Combines LLM, Embedding, and CSV loading to create a vector index for the documents.

**create_search_query() Function**
```python
def create_serach_query():
  #function to ask query from csv file

# create a query engine and query

  vector_index=create_chunk()
  query_engine = vector_index.as_query_engine(response_mode="compact")

  response = query_engine.query("Can you bring some insights from the provided data ?")

  pprint_response(response, show_source=True)
```

Executes a search query using the previously created vector index.
The code sets up an environment to load language and embedding models, process a CSV file, create a database index, and execute a sample query to derive insights from the stored data. Each function serves a specific purpose in this data processing and querying pipeline.


