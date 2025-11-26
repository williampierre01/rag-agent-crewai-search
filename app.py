import gradio as gr
import os
import spaces
import torch
import gc
from smolagents import CodeAgent, TransformersModel, Tool, DuckDuckGoSearchTool

# Imports de RAG
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ==============================================================================
#  ESTADO GLOBAL
# ==============================================================================
GLOBAL_RETRIEVER = None
GLOBAL_MODEL = None 

# ==============================================================================
#  1. FERRAMENTA DE PDF
# ==============================================================================
class PDFSearchTool(Tool):
    name = "search_pdf"
    description = "Search for specific information within the uploaded PDF document."
    inputs = {
        "query": {
            "type": "string",
            "description": "The specific question or keywords to search for in the PDF."
        }
    }
    output_type = "string"

    def forward(self, query: str) -> str:
        global GLOBAL_RETRIEVER
        if GLOBAL_RETRIEVER is None:
            return "Error: No PDF uploaded."
        
        try:
            docs = GLOBAL_RETRIEVER.invoke(query)
            if not docs:
                return "No relevant information found in the PDF."
            return "\n\n".join([f"--- Content ---\n{doc.page_content}" for doc in docs])
        except Exception as e:
            return f"Error searching PDF: {str(e)}"

# ==============================================================================
#  2. CARREGAMENTO DO MODELO
# ==============================================================================
def load_model():
    global GLOBAL_MODEL
    if GLOBAL_MODEL is not None:
        return GLOBAL_MODEL
        
    print("--- CARREGANDO QWEN 2.5 NA GPU ---")
    GLOBAL_MODEL = TransformersModel(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        device_map="auto",
        torch_dtype=torch.bfloat16,
        load_in_4bit=True,
        max_new_tokens=1024
    )
    return GLOBAL_MODEL

# ==============================================================================
#  3. AGENTES (SOLUÇÃO SEM ManagedAgent)
# ==============================================================================

# --- CLASSE ESPECIAL: AGENTE COMO FERRAMENTA ---
# Isso substitui o ManagedAgent. Envolvemos um agente dentro de uma Tool.
class AgentAsTool(Tool):
    def __init__(self, agent, name, description):
        self.agent = agent
        self.name = name
        self.description = description
        self.inputs = {
            "task": {
                "type": "string",
                "description": "The task or question to delegate to this agent."
            }
        }
        self.output_type = "string"
        super().__init__()

    def forward(self, task: str) -> str:
        try:
            return str(self.agent.run(task))
        except Exception as e:
            return f"Error from sub-agent: {str(e)}"

def get_multi_agent_system():
    model = load_model()

    # 1. Agente Pesquisador (PDF + Web)
    retriever = CodeAgent(
        tools=[PDFSearchTool(), DuckDuckGoSearchTool()],
        model=model,
        add_base_tools=False,
        description="A meticulous analyst that searches PDF and Web."
    )

    # 2. Transformamos o Pesquisador em uma Ferramenta
    # Assim o Gerente pode "usá-lo" como se fosse uma calculadora ou busca.
    researcher_tool = AgentAsTool(
        agent=retriever,
        name="ask_researcher",
        description="Use this tool to ask the Researcher Agent to find information. Give it a detailed question."
    )

    # 3. Agente Gerente (Synthesizer)
    # Ele recebe a ferramenta 'ask_researcher'
    
    system_prompt = """
    You are a Senior Response Synthesizer.
    Your goal is to answer the user's question clearly.
    
    You have a powerful tool called 'ask_researcher'.
    ALWAYS delegate the research to 'ask_researcher' first.
    
    If the researcher returns information, summarize it nicely.
    If the researcher fails, apologize.
    """
    
    manager = CodeAgent(
        tools=[researcher_tool], # O agente virou ferramenta aqui
        model=model,
        add_base_tools=False
    )
    
    return manager, system_prompt

# ==============================================================================
#  4. INDEXAÇÃO
# ==============================================================================
@spaces.GPU
def build_vector_store(pdf_path):
    global GLOBAL_RETRIEVER
    print(f"[RAG] Indexando: {pdf_path}")
    
    try:
        torch.cuda.empty_cache()
        gc.collect()

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)

        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embedding_model,
            collection_name="pdf_rag_store",
            persist_directory=None
        )
        
        GLOBAL_RETRIEVER = vectorstore.as_retriever(search_kwargs={"k": 3})
        load_model()
        return True
    except Exception as e:
        print(f"[ERRO RAG] {e}")
        return False

# ==============================================================================
#  5. CHAT FUNCTION
# ==============================================================================
@spaces.GPU(duration=120)
def chat_function(message, history):
    if not message: return history
    if history is None: history = []
    
    history.append([message, "🤖 Gerente delegando para Pesquisador..."])
    yield history

    try:
        manager, prompt = get_multi_agent_system()
        
        full_msg = f"{prompt}\n\nUSER QUESTION: {message}"
        
        response = manager.run(full_msg)
        
        history[-1] = [message, str(response)]
        
    except Exception as e:
        error_msg = f"Erro na execução: {str(e)}"
        print(error_msg)
        history[-1] = [message, error_msg]
    
    yield history

# ==============================================================================
#  INTERFACE
# ==============================================================================
def process_pdf(file_obj):
    if not file_obj: return "Sem arquivo."
    try:
        path = file_obj.name if hasattr(file_obj, 'name') else file_obj
        if build_vector_store(path):
            return "PDF Indexado! Agentes Prontos."
        else:
            return "Erro ao indexar PDF."
    except Exception as e: return f"Erro: {e}"

with gr.Blocks(title="SmolAgents Multi v2") as demo:
    gr.Markdown("# 🧠 Multi-Agent RAG (Stable Version)")
    gr.Markdown("Manager ➡️ Researcher (PDF + Web)")
    
    with gr.Row():
        with gr.Column(scale=1):
            upl = gr.File(label="Upload PDF")
            status = gr.Markdown("Aguardando arquivo...")
        
        with gr.Column(scale=4):
            chat = gr.Chatbot(height=600)
            msg = gr.Textbox(label="Pergunta")
    
    upl.change(process_pdf, upl, status)
    msg.submit(chat_function, [msg, chat], [chat])

if __name__ == "__main__":
    demo.launch()
