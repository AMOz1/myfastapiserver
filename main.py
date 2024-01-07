from fastapi import FastAPI
from txtai.embeddings import Embeddings

app = FastAPI()

# Initialize txtai embeddings instance
embeddings = Embeddings("app.yml")

@app.get("/search")
def search(query: str):
    return embeddings.search(query, 1)
