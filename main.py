import bs4
from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from fastapi import FastAPI
from langserve import add_routes
from langchain.vectorstores import FAISS
import uvicorn
import os

load_dotenv()

# ingest document from website. This loader can be replaced by other loader to load 
# other types of documents. ex: pdf in a folder
#  def ingest_docs():
#  loader = WebBaseLoader(
#    web_paths=("https://medicalxpress.com/news/2024-01-ai-tool-precision-# # 
# pathology-cancer.html",),
#    bs_kwargs=dict(
#        parse_only=bs4.SoupStrainer(
#            class_=("news-article")
#        )
#    ),
#  )
loader = DirectoryLoader(
    './docs',
    glob="**/*",
    show_progress=False,
    use_multithreading=True
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)

vectorstore = FAISS.from_documents(documents=chunks, embedding=OpenAIEmbeddings())
vectorstore.save_local("faiss_vector_db")
print("Document successfully ingested")

if not os.path.isdir("faiss_vector_db"):
    print("Folder does not exist.")
    ingest_docs()

vectorstore = FAISS.load_local("faiss_vector_db", OpenAIEmbeddings())

retriever = vectorstore.as_retriever()

prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

add_routes(
    app,
    rag_chain,
    path="/knowledge",
)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
