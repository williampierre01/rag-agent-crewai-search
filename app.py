import gradio as gr
import os
import spaces
import torch
import gc
from smolagents import CodeAgent, TransformersModel, Tool, ManagedAgent, DuckDuckGoSearchTool

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
#  2. CARREGAMENTO DO MODELO (SINGLETON)
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
#  3. CRIAÇÃO DOS AGENTES (PDF + WEB)
# ==============================================================================
def get_multi_agent_system():
    model = load_model()

    # --- INSTANCIAR FERRAMENTAS ---
    pdf_tool = PDFSearchTool()
    web_tool = DuckDuckGoSearchTool() # Ferramenta de Busca Web Gratuita

    # --- AGENTE 1: RETRIEVER AGENT ---
    # Agora ele tem DUAS ferramentas: PDF e Web.
    retriever = CodeAgent(
        tools=[pdf_tool, web_tool],
        model=model,
        name="retriever_agent",
        description="""You are a meticulous analyst. 
        Your goal is to retrieve the most relevant information.
        IMPORTANT:
        1. ALWAYS try to use the 'search_pdf' tool FIRST.
        2. ONLY if the PDF tool returns no results or insufficient info, then use the 'web_search' tool.
        """,
        add_base_tools=False
    )

    managed_retriever = ManagedAgent(
        agent=retriever,
        name="retriever_agent",
        description="Call this agent to retrieve information. It can check the PDF and the Web."
    )

    # --- AGENTE 2: RESPONSE SYNTHESIZER (MANAGER) ---
    system_prompt_synthesizer = """
    You are the 'response_synthesizer_agent'.
    
    BACKSTORY:
    You're a skilled communicator with a knack for turning complex information into clear and concise responses.
    
    GOAL:
    Synthesize the retrieved information into a concise and coherent response based on the user query.
    
    INSTRUCTIONS:
    1. Delegate the research to 'retriever_agent'.
    2. If the retriever finds information (from PDF or Web), summarize it clearly.
    3. If the retriever fails completely (nothing in PDF and nothing on Web), respond with: "I'm sorry, I couldn't find the information you're looking for."
    """

    synthesizer_agent = CodeAgent(
        tools=[], 
        managed_agents=[managed_retriever],
        model=model,
        add_base_tools=False
    )
    
    return synthesizer_agent, system_prompt_synthesizer

# ==============================================================================
#  4. INDEXAÇÃO (RAG)
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
    
    history.append([message, "🤖 Buscando no PDF (e Web se necessário)..."])
    yield history

    try:
        synthesizer, persona_prompt = get_multi_agent_system()
        
        full_instruction = f"{persona_prompt}\n\nUSER QUERY: {message}"
        
        response = synthesizer.run(full_instruction)
        
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
            return "PDF Indexado! Agentes com Web Search prontos."
        else:
            return "Erro ao indexar PDF."
    except Exception as e: return f"Erro: {e}"

with gr.Blocks(title="SmolAgents PDF+Web") as demo:
    gr.Markdown("# 🧠 Multi-Agent RAG (PDF + Web)")
    gr.Markdown("**Retriever** (Prioriza PDF, usa Web se falhar) ➡️ **Synthesizer** (Responde)")
    
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
