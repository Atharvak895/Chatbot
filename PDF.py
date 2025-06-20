from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables import RunnableParallel,RunnablePassthrough,RunnableLambda
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

loader = PyPDFLoader("Waterrr.pdf")
docs = loader.load()
spliter = RecursiveCharacterTextSplitter(chunk_size = 500,chunk_overlap = 50)
chunks = spliter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vec = FAISS.from_documents(chunks,embeddings)
retriver  = vec.as_retriever()
retriver

llm = ChatGroq(model="llama-3.1-8b-instant")
prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided context.
      please be polite with the user.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']   
)


def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text
parallel_chain = RunnableParallel({
    'context': retriver | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})
parser = StrOutputParser()
main_chain = parallel_chain | prompt | llm | parser
#print(main_chain.invoke('importance of water'))

while True:
    user_input = input("You :")
    if user_input.lower() in ["exit","quit","e","q"]:
        print("Exiting chat")
        break
    result = main_chain.invoke(user_input)
    print("Bot:", result)