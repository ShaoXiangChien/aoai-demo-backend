import logging
import azure.cosmos.documents as documents
import azure.cosmos.cosmos_client as cosmos_client
import azure.cosmos.exceptions as exceptions
from azure.cosmos.partition_key import PartitionKey
import pandas as pd
import numpy as np
import os
import json
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
from pydantic import BaseModel
from typing import Optional
from fastapi import FastAPI, HTTPException, Path, Query
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()
openai.api_type = 'azure'
openai.api_version = "2022-12-01"
openai.api_key = os.environ.get('openai_key')
openai.api_base = os.environ.get('openai_base')

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
HOST = os.environ.get('host')
MASTER_KEY = os.environ.get('master_key')
DATABASE_ID = os.environ.get('database_id')
CONTAINER_ID = os.environ.get('container_id')

client = cosmos_client.CosmosClient(
    HOST, {'masterKey': MASTER_KEY}, user_agent="CosmosDBPythonQuickstart", user_agent_overwrite=True)
db = client.get_database_client(DATABASE_ID)
container = db.get_container_client(CONTAINER_ID)
docs = pd.read_csv("./COVID-FAQ_qnas.csv")
embeddings = np.load("./question_embeddings.npy").tolist()

"""
create a model class with BaseModel that can store the following json data:
{"id":"chatcmpl-6v7mkQj980V1yBec6ETrKPRqFjNw9",
"object":"chat.completion","created":1679072642,
"model":"gpt-35-turbo",
"usage":{"prompt_tokens":58,
"completion_tokens":68,
"total_tokens":126},
"choices":[{"message":{"role":"assistant",
"content":"Yes, other Azure Cognitive Services also support customer managed keys. Azure Cognitive Services offer multiple options for customers to manage keys, such as using Azure Key Vault, customer-managed keys in Azure Key Vault or customer-managed keys through Azure Storage service. This helps customers ensure that their data is secure and access to their services is controlled."},"finish_reason":"stop","index":0}]}
"""


class ChatCompletion(BaseModel):
    id: str
    model: Optional[str]
    user_id: str
    usage: Optional[dict]
    role: str
    content: str


class RequestBody(BaseModel):
    data: str


@app.get("/items/{user_id}/{n}")
def get_last_n_item(user_id: str, n: int) -> list[ChatCompletion]:
    query = "SELECT * FROM c WHERE c.user_id LIKE '{0}%' ORDER BY c._ts DESC".format(
        user_id)

    logging.info("Executing query: {}".format(query))
    items = list(container.query_items(
        query=query,
        enable_cross_partition_query=True
    ))
    logging.info("Found {} items".format(len(items)))
    return [{"role": item['role'], "content": item['content']} for item in (reversed(items[:n]) if len(items) >= n else items)]


@app.get("/item")
def get_item(user_id: str = Query(), item_id: str = Query()) -> ChatCompletion:
    item = container.read_item(item=item_id, partition_key=user_id)
    return item


@app.post("/")
def create_item(item: ChatCompletion):
    container.create_item(body=item.dict())


# @app.get("/search")
# def search_docs(user_query: str = Query()):
#     embedding = get_embedding(
#         user_query,
#         engine="text-similarity-curie-001"
#     )
#     docs['similarities'] = [cosine_similarity(
#         emb, embedding) for emb in embeddings]
#     return docs.sort_values(by='similarities', ascending=False).head(3)['Answer'].to_list()


@app.post("/search")
def search_docs_emb(body: RequestBody):
    embedding = json.loads(body.data).get('data')[0].get('embedding')

    docs['similarities'] = [cosine_similarity(
        emb, embedding) for emb in embeddings]
    return docs.sort_values(by='similarities', ascending=False).head(3)['Answer'].to_list()
