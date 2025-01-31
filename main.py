from fastapi import FastAPI
from pydantic import BaseModel
from vector_store import load_faiss_index
from langchain.chains import RetrievalQA
from langchain.chat_models import AzureChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Load FAISS index
vector_store = load_faiss_index()

print(" dep : "+os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"))
print(" end : "+os.getenv("AZURE_OPENAI_ENDPOINT"))
print(" key : "+os.getenv("AZURE_OPENAI_API_KEY"))
# Load Azure OpenAI model
llm = AzureChatOpenAI(
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY")
)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_store.as_retriever()
)

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(request: QuestionRequest):
    """Query FAISS index and generate an answer."""
    answer = qa_chain.run(request.question)
    return {"question": request.question, "answer": answer}
