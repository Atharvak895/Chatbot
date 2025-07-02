from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.environ['GROQ_API_KEY']

app = FastAPI()

loader = PyPDFLoader("Waterrr.pdf")
docs = loader.load()
spliter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = spliter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vec = FAISS.from_documents(chunks, embeddings)
retriever = vec.as_retriever()

llm = ChatGroq(model="llama-3.1-8b-instant")
prompt = PromptTemplate(
    template="""
    You are a helpful assistant.
    Answer ONLY from the provided context.
    Please be polite with the user.
    If the context is insufficient, just say you don't know.

    {context}
    Question: {question}
    """,
    input_variables=["context", "question"]
)

def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})
parser = StrOutputParser()
main_chain = parallel_chain | prompt | llm | parser

class Query(BaseModel):
    question: str

@app.get("/")
async def root():
    return {"message": "Chatbot API is live!"}

@app.post("/chat")
async def chat_with_bot(query: Query):
    response = main_chain.invoke(query.question)
    return {"response": response}
