import os
import logging
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"

# these three lines swap the stdlib sqlite3 lib with the pysqlite3 package
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# Configure logging
logging.basicConfig(
    # filename="app.log",  # Log file
    stream=sys.stdout,
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,  # Log level (can also be DEBUG for more verbosity)
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

import torch
import chromadb
from chromadb.utils import embedding_functions


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
chroma_client = chromadb.PersistentClient(path="./data")

def register_collection(collection_name):
    # append to new line collection_name as string to COLLECTIONS.txt file
    logger.info(f"Writing collection name to collections list....")
    with open("COLLECTIONS.txt", "a") as f:
        f.write(collection_name + "\n")
    logger.info(f"Completed adding collection name to collections list....")


def create_vector_db(docs, model_name, collection_name):

    # create or load the vector store
    try:
        if len(chroma_client.list_collections()) > 0 and collection_name in [
            collct.name for collct in chroma_client.list_collections()
        ]:
            logger.info(f"Collection Already exisit so getting that collection.....")
            print( [
            collct.name for collct in chroma_client.list_collections()
        ])
            print(collection_name)
            collection = chroma_client.get_collection(name=collection_name)
            return {'ERROR': f"There already exists a collection with {collection_name}. Please give a new collection name"}
        else:
            # create the open-source embedding function
            logger.info("Creating new collection for the uploaded file.....")
            embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name,
                                                                                        device = device)
            collection = chroma_client.create_collection(name=collection_name, embedding_function = embedding_function) 
            register_collection(collection_name)
            
            num_ids = collection.count()
            num_docs = len(docs)  
            logger.info("Adding Documents to newly created collection......")  
            collection.add(
                documents = [doc.page_content for doc in docs],
                ids = [f'id_{i}' for i in range(num_ids, num_ids + num_docs)],
                metadatas=  [doc.metadata for doc in docs])

    except Exception as e:
        raise e


def load_local_db(collection_name):
    collection = chroma_client.get_collection(name=collection_name)
    return collection


