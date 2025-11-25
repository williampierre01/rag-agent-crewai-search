import gradio as gr
import os
import spaces
from crewai import Agent, Crew, Process, Task
from crewai.tools import BaseTool

# RAG Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Integração Oficial LiteLLM/LangChain
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# ==============================================================================
#  CONFIGURAÇÕES DO SISTEMA
# ==============================================================================
# Desativa telemetria e validações externas para evitar chamadas de rede
os.environ["OPENAI_API_KEY"] = "sk-fake-key-local"
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["LITELLM_LOG"] = "ERROR" # Reduz logs do LiteLLM

# ==============================================================================
#  GLOBAL STATE
# ==============================================================================
VECTOR_DB_RETRIEVER = None
GLOBAL_LLM = None

# ==============================================================================
#  1. BANCO VETORIAL (RAG)
# ==============================================================================
@spaces.GPU
def build_vector_store(pdf_path):
    global VECTOR_DB_RETRIEVER
    print(f"[RAG] Indexando: {pdf_path}")
    
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embedding_model,
            collection_name="pdf_rag_store",
            persist_directory=None
        )
        
        VECTOR_DB_RETRIEVER = vectorstore.as_retriever(search_kwargs={"k": 4})
        return True
    except Exception as e:
        print(f"[ERRO RAG] {e}")
        return False

# ==============================================================================
#  2. FERRAMENTA CUSTOMIZADA
# ==============================================================================
class PDFRagTool(BaseTool):
    name: str = "SearchPDF"
    description: str = "Search the PDF content. Input is the search query."

    def _run(self, query: str) -> str:
        if VECTOR_DB_RETRIEVER is None: return "Error: No PDF loaded."
        try:
            docs = VECTOR_DB_RETRIEVER.invoke(query)
            return "\n".join([d.page_content for d in docs])
        except: return "No info found."

# ==============================================================================
#  3. CARREGAMENTO DO MODELO (INTEGRAÇÃO CHAT)
# ==============================================================================
def load_llm():
    global GLOBAL_LLM
    if GLOBAL_LLM: return GLOBAL_LLM

    print("--- CARREGANDO MODELO PHI-3.5 ---")
    model_name = "microsoft/Phi-3.5-mini-instruct"
    
    # Configuração de Quantização (4-bit) para economizar memória
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True
    )

    # Pipeline de Texto Puro
    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        return_full_text=False,
        do_sample=True,
        temperature=0.1,
    )

    # Wrapper do LangChain
    hf_pipeline = HuggingFacePipeline(pipeline=text_pipeline)

    # AQUI ESTÁ A MÁGICA DO LITELLM:
    # Usamos ChatHuggingFace para converter o modelo local em um "ChatModel"
    # que o CrewAI/LiteLLM entende nativamente.
    GLOBAL_LLM = ChatHuggingFace(llm=hf_pipeline, tokenizer=tokenizer)
    
    print("--- MODELO CARREGADO ---")
    return GLOBAL_LLM

# ==============================================================================
#  4. AGENTES
# ==============================================================================
def create_agents_and_tasks(user_query):
    # Carrega o modelo
    llm = load_llm()
    
    tools = [PDFRagTool()] if VECTOR_DB_RETRIEVER else []

    researcher = Agent(
        role="Researcher",
        goal="Find facts in the PDF.",
        backstory="Analytical researcher.",
        tools=tools,
        llm=llm,
        verbose=True,
        allow_delegation=False, # Importante para evitar chamadas complexas
        cache=False # Evita erros de cache do LiteLLM
    )

    writer = Agent(
        role="Writer",
        goal="Write a final answer.",
        backstory="Technical writer.",
        tools=[], 
        llm=llm,
        verbose=True,
        allow_delegation=False,
        cache=False
    )

    task1 = Task(
        description=f"Search PDF for: '{user_query}'. Return list of facts.",
        expected_output="List of facts.",
        agent=researcher
    )

    task2 = Task(
        description=f"Answer '{user_query}' based on facts.",
        expected_output="Final text.",
        agent=writer
    )

    return Crew(
        agents=[researcher, writer],
        tasks=[task1, task2],
        process=Process.sequential,
        verbose=True,
        memory=False, 
        cache=False,  
        embedder={    
             "provider": "huggingface",
             "config": {"model": "sentence-transformers/all-MiniLM-L6-v2"}
        }
    )

# ==============================================================================
#  5. INTERFACE
# ==============================================================================
def process_pdf(file_obj):
    if not file_obj: return "Sem arquivo."
    try:
        path = file_obj.name if hasattr(file_obj, 'name') else file_obj
        if build_vector_store(path):
            return "PDF Indexado!"
        else:
            return "Erro na indexação."
    except Exception as e: return f"Erro: {e}"

@spaces.GPU(duration=120)
def chat_function(message, history):
    if not message: return history
    if history is None: history = []
    
    history.append([message, "🤖 Rodando Agentes (Via ChatHuggingFace)..."])
    yield history

    try:
        crew = create_agents_and_tasks(message)
        result = crew.kickoff()
        history[-1] = [message, str(result.raw)]
    except Exception as e:
        # Mostra o erro real se acontecer
        error_message = f"Erro: {str(e)}"
        print(f"ERRO FATAL: {error_message}")
        history[-1] = [message, error_message]
    
    yield history

with gr.Blocks(title="RAG Final LiteLLM") as demo:
    gr.Markdown("# 🛡️ RAG Local (Standard Integration)")
    with gr.Row():
        upl = gr.File(label="PDF")
        st = gr.Markdown("...")
    chat = gr.Chatbot(height=550)
    msg = gr.Textbox(label="Pergunta")
    
    upl.change(process_pdf, upl, st)
    msg.submit(chat_function, [msg, chat], [chat])

if __name__ == "__main__":
    demo.launch()
