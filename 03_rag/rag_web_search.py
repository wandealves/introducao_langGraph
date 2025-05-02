import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langchain import hub
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, StateGraph, START
from typing import List
from typing_extensions import TypedDict
from pprint import pprint

# -----------------------------
# Carregar variáveis de ambiente
# -----------------------------
load_dotenv()
os.environ['TAVILY_API_KEY'] = os.getenv("TAVILY_API_KEY")

# -----------------------------
# Carregamento e indexação de documentos
# -----------------------------
urls = [
    "https://www.revistas.usp.br/revbiologia/article/view/217747",
    "https://www.revistas.usp.br/revbiologia/article/view/206105",
    "https://www.revistas.usp.br/revbiologia/article/view/219858",
]

raw_docs = [doc for url in urls for doc in WebBaseLoader(url).load()]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250, chunk_overlap=0)
doc_chunks = text_splitter.split_documents(raw_docs)

vectorstore = Chroma.from_documents(
    documents=doc_chunks,
    collection_name="rag-chroma",
    embedding=OpenAIEmbeddings(),
)
retriever = vectorstore.as_retriever()

llm_main = ChatOpenAI(model_name="gpt-4.1-nano", temperature=0)
llm_advanced = ChatOpenAI(model="gpt-4.1-nano", temperature=0)

rag_prompt = hub.pull("rlm/rag-prompt")
rag_chain = rag_prompt | llm_main | StrOutputParser()

class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Os documentos são relevantes à pergunta, 'sim' ou 'não'")

grade_prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um avaliador que verifica a relevância de um documento recuperado em relação a uma pergunta do usuário. Responda apenas em português do Brasil."),
    ("human", "Documento recuperado:\n\n{document}\n\nPergunta do usuário: {question}"),
])
retrieval_grader = grade_prompt | llm_advanced.with_structured_output(GradeDocuments)

re_write_prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um reescritor de perguntas que transforma a entrada em uma versão melhor para busca. Responda em português do Brasil."),
    ("human", "Pergunta inicial:\n\n{question}\n\nFormule uma pergunta melhorada."),
])
question_rewriter = re_write_prompt | llm_advanced | StrOutputParser()

web_search_tool = TavilySearchResults(k=3)

class GraphState(TypedDict):
    question: str
    generation: str
    web_search: str
    documents: List[Document]
    full_context: str

def recuperar(state):
    print(">>> Recuperar documentos")
    question = state["question"]
    documents = retriever.get_relevant_documents(question)
    return {"documents": documents, "question": question}

def avaliar_documentos(state):
    print(">>> Avaliar relevância dos documentos")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    for doc in documents:
        score = retrieval_grader.invoke({"question": question, "document": doc.page_content})
        if score.binary_score.lower() == "sim":
            filtered_docs.append(doc)

    web_search = "Sim" if len(filtered_docs) < 1 else "Não"
    return {"documents": filtered_docs, "question": question, "web_search": web_search}

def transformar_pergunta(state):
    print(">>> Reescrever pergunta")
    question = state["question"]
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": state["documents"], "question": better_question}

def busca_web(state):
    print(">>> Buscar na Web")
    question = state["question"]
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    state["documents"].append(Document(page_content=web_results))
    return {"documents": state["documents"], "question": question}

def gerar_resposta(state):
    print(">>> Gerar resposta")
    return {
        "documents": state["documents"],
        "question": state["question"],
        "generation": rag_chain.invoke({"context": state["documents"], "question": state["question"]}),
        "web_search": state.get("web_search", "Não"),
        "full_context": "\n\n".join([doc.page_content for doc in state["documents"]])
    }

def decidir_geracao(state):
    if state["web_search"] == "Sim":
        return "transformar_pergunta"
    return "gerar_resposta"

# -----------------------------
# Construção do grafo
# -----------------------------
workflow = StateGraph(GraphState)
workflow.add_node("recuperar", recuperar)
workflow.add_node("avaliar_documentos", avaliar_documentos)
workflow.add_node("transformar_pergunta", transformar_pergunta)
workflow.add_node("busca_web", busca_web)
workflow.add_node("gerar_resposta", gerar_resposta)

workflow.add_edge(START, "recuperar")
workflow.add_edge("recuperar", "avaliar_documentos")
workflow.add_conditional_edges("avaliar_documentos", decidir_geracao, {
    "transformar_pergunta": "transformar_pergunta",
    "gerar_resposta": "gerar_resposta"
})
workflow.add_edge("transformar_pergunta", "busca_web")
workflow.add_edge("busca_web", "gerar_resposta")
workflow.add_edge("gerar_resposta", END)

app = workflow.compile()

# -----------------------------
# Execução
# -----------------------------
inputs = {"question": "Prevalência e detecção de resistência de Staphylococcus"}
for output in app.stream(inputs):
    for key, value in output.items():
        pprint(f"Nó '{key}':")
    pprint("\n---\n")

# Resposta final
pprint(value["full_context"])
