from typing import Optional
import torch
import logging
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query, File, UploadFile,Form
from fastapi.middleware.cors import CORSMiddleware
from llama_index.llms.groq import Groq

from load_data import load_split_pdf_file, load_split_html_file, initialize_splitter
from vector_db import create_vector_db, load_local_db
from prompts import create_prompt
from utils import read_file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fake_output(x: float):
    return "Answer to this query is 42"

ml_models = {}
db_name = {}

text_splitter = initialize_splitter(chunk_size = 1000, chunk_overlap = 100)
vector_db_model_name = "all-MiniLM-L6-v2"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    yield
    # Clean up the ML models and release the resources
    ml_models.clear()

# Configure logging
logging.basicConfig(
    # filename="app.log",  # Log file
    stream=sys.stdout,
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,  # Log level (can also be DEBUG for more verbosity)
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Create logger object
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG_APP",
    description="Retrival Augmented Generation APP which let's user upload a file and get the answer for the question using LLMs",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
def index():
    return {"message": "Hello World"}



# the model initialized when the app gets loaded but we can configure it if we want
@app.get("/init_llm")
def init_groq_api(model_name: str = Query('llama3-8b-8192', description="Name of groq model to use as LLM"),
                  api_key: str = Query('sk-xxxx',description="Provide the api key to your groq account")):
    
    llm = Groq(model=model_name,api_key=api_key)
    logger.info("Gorq API is read and model initialized......")
    ml_models["answer_to_query"] = llm
    return {"message": "LLM initialized"}

@app.post("/upload")
def upload_file(file: UploadFile = File(...), collection_name : Optional[str] = Form(...)):
    try:
        print(collection_name)

        logger.info("Reading contents from Uploaded File")
        contents = file.file.read()
        with open(f'./data/{file.filename}', 'wb') as f:
            f.write(contents)
        logger.info(f"Loaded content and file saved locally...")
    except Exception:
        logger.error(f"Error in reading Uploaded file in local......")
        return {"message": "There was an error uploading the file","status":"Uploading File Error"}
    finally:
        file.file.close()
    
    if file.filename.endswith('.pdf'):
        logger.info(f"Loading and splitting file into chunks........")
        data = load_split_pdf_file(f'./data/{file.filename}', text_splitter)
    elif file.filename.endswith('.html'):
        logger.info(f"Loading and splitting file into chunks........")
        data = load_split_html_file(f'./data/{file.filename}', text_splitter)
    else:
        return {"message": "Only pdf and html files are supported","status":"File Format Error"}
    
    logger.info(f"Creating VectorDB for Uploaded file.......")
    db = create_vector_db(data, vector_db_model_name, collection_name)
    logger.info(f"Completed VectorDB creation.......")

    if db is None:
        return {"message": f"Successfully uploaded {file.filename} to {collection_name}", 
                "num_splits" : len(data),"status":"collection_created"}
    else:
        return db


@app.get("/query")
def query(query : str, n_results : Optional[int] = 2, collection_name : Optional[str] = "test_collection"):
    try:
        collection_list = read_file('COLLECTIONS.txt')
        collection_list = collection_list.split("\n")[:-1]
    except Exception:
        return {"message": "No collections found uplaod some documents first"}

    if collection_name not in collection_list:
        return {"message": f"There is no collection with name {collection_name}",
                "available_collections" : collection_list}
    collection = load_local_db(collection_name)
    results = collection.query(query_texts=[query], n_results = n_results)
    prompt = create_prompt(query, results)
    output = ml_models["answer_to_query"].complete(prompt)
    return {"message": f"Query is {query}",
        "relavent_docs" : results,
        "llm_output" : output}
    


if __name__ == "__main__":
    pass

