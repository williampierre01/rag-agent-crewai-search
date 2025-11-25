import gradio as gr
import os
import spaces
from crewai import Agent, Crew, Process, Task
from crewai.tools import BaseTool

# RAG / Vector DB Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# LLM Imports
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from langchain_community.llms import HuggingFacePipeline

# ==============================================================================
#  CONFIGURAÇÕES DE SEGURANÇA (Anti-OpenAI)
# ==============================================================================
os.environ["OPENAI_API_KEY"] = "NA"
os.environ["OTEL_SDK_DISABLED"] = "true"

# ==============================================================================
#  GLOBAL STATE
# ==============================================================================
VECTOR_DB_RETRIEVER = None

# ==============================================================================
#  1. PREPARAÇÃO DO BANCO VETORIAL (RAG)
# ==============================================================================
@spaces.GPU
def build_vector_store(pdf_path):
    global VECTOR_DB_RETRIEVER
    print(f"[RAG] Processando PDF: {pdf_path}")
    
    # Carrega e divide o PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    # Chunk size ajustado para o Phi-3.5 (não perder contexto)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    # Embeddings Locais
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Cria Banco Vetorial na Memória
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_model,
        collection_name="pdf_knowledge_base",
        persist_directory=None
    )
    
    VECTOR_DB_RETRIEVER = vectorstore.as_retriever(search_kwargs={"k": 4})
    print("[RAG] Banco pronto.")
    return True

# ==============================================================================
#  2. FERRAMENTA DE BUSCA NO PDF
# ==============================================================================
class PDFRagTool(BaseTool):
    name: str = "SearchPDF"
    description: str = "Search for information inside the PDF. Input: The query string."

    def _run(self, query: str) -> str:
        global VECTOR_DB_RETRIEVER
        if VECTOR_DB_RETRIEVER is None:
            return "Error: No PDF uploaded."
        try:
            docs = VECTOR_DB_RETRIEVER.invoke(query)
            # Retorna o texto encontrado
            return "\n".join([d.page_content for d in docs])
        except Exception:
            return "No information found."

# ==============================================================================
#  3. MODELO LLM (Local)
# ==============================================================================
global_model = None
global_tokenizer = None

def initialize_model():
    global global_model, global_tokenizer
    if global_model: return global_model, global_tokenizer

    model_name = "microsoft/Phi-3.5-mini-instruct"
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=bnb_config,
    )
    global_tokenizer = tokenizer
    global_model = model
    return global_model, global_tokenizer

def load_llm():
    model, tokenizer = initialize_model()
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.1,
        return_full_text=False
    )
    return HuggingFacePipeline(pipeline=pipe)

# ==============================================================================
#  4. AGENTES E TAREFAS (Configuração de 2 Agentes)
# ==============================================================================
def create_agents_and_tasks(user_query):
    
    llm = load_llm()
    
    # Apenas o Agente 1 recebe a ferramenta de busca
    tools = [PDFRagTool()] if VECTOR_DB_RETRIEVER else []

    # --- AGENTE 1: O CAÇADOR DE FATOS ---
    researcher = Agent(
        role="Senior Researcher",
        goal="Extract precise information from the PDF.",
        backstory="You are an expert at finding facts in documents.",
        tools=tools,
        llm=llm,
        verbose=True,
        allow_delegation=False, # Importante: Sem delegação
        cache=False             # Importante: Sem cache
    )

    # --- AGENTE 2: O ESCRITOR ---
    writer = Agent(
        role="Technical Writer",
        goal="Write a clear, concise answer based on the Researcher's findings.",
        backstory="You write easy-to-understand responses.",
        tools=[], # Nenhuma ferramenta, ele apenas processa texto
        llm=llm,
        verbose=True,
        allow_delegation=False, # Importante: Sem delegação
        cache=False             # Importante: Sem cache
    )

    # --- TAREFAS ---
    task1 = Task(
        description=f"Search the PDF for information regarding: '{user_query}'. Return the raw facts found.",
        expected_output="A list of key facts and quotes from the text.",
        agent=researcher
    )

    task2 = Task(
        description=f"Using the context provided by the Researcher, write a final answer for: '{user_query}'",
        expected_output="A well-formatted text response.",
        agent=writer
    )

    # --- CREW ---
    return Crew(
        agents=[researcher, writer],
        tasks=[task1, task2],
        process=Process.sequential, # Garante: Agente 1 -> Agente 2
        verbose=True,
        memory=False,      # Desliga OpenAI Memory
        cache=False,       # Desliga OpenAI Cache
        manager_llm=None,  # Garante que não tem Manager
        embedder={         # Redundância de segurança
             "provider": "huggingface",
             "config": {"model": "sentence-transformers/all-MiniLM-L6-v2"}
        }
    )

# ==============================================================================
#  5. INTERFACE GRADIO
# ==============================================================================
def process_pdf(file_obj):
    if not file_obj: return "Sem arquivo."
    try:
        path = file_obj.name if hasattr(file_obj, 'name') else file_obj
        build_vector_store(path)
        return "PDF Indexado com Sucesso!"
    except Exception as e:
        return f"Erro: {e}"

@spaces.GPU(duration=120)
def chat_function(message, history):
    if not message: return history
    if history is None: history = []
    
    # Feedback visual para o usuário
    history.append([message, "🤖 Agente 1 Pesquisando... -> ✍️ Agente 2 Escrevendo..."])
    yield history

    try:
        crew = create_agents_and_tasks(message)
        result = crew.kickoff()
        history[-1] = [message, str(result.raw)]
    except Exception as e:
        history[-1] = [message, f"Erro: {str(e)}"]
    
    yield history

with gr.Blocks(title="Multi-Agent RAG Local") as demo:
    gr.Markdown("# 🤖🤖 Multi-Agent RAG (Pesquisador + Escritor)")
    
    with gr.Row():
        upl = gr.File(label="Upload PDF")
        st = gr.Markdown("Aguardando PDF...")
    
    chat = gr.Chatbot(height=600)
    msg = gr.Textbox(label="Pergunta")
    
    upl.change(process_pdf, upl, st)
    msg.submit(chat_function, [msg, chat], [chat])

if __name__ == "__main__":
    demo.launch()
