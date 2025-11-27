import gradio as gr
import os
import spaces
import torch
import gc
import traceback
from smolagents import CodeAgent, Tool, Model, ChatMessage

# Imports de RAG
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Imports Transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

# ==============================================================================
#  ESTADO GLOBAL
# ==============================================================================
GLOBAL_RETRIEVER = None
GLOBAL_ENGINE = None 

# ==============================================================================
#  1. ENGINE CUSTOMIZADA (CORREÇÃO DE TIPOS LIST vs STR)
# ==============================================================================
class LocalQwenEngine(Model):
    def __init__(self, pipeline_obj):
        super().__init__()
        self.pipeline = pipeline_obj
        self.tokenizer = pipeline_obj.tokenizer

    def get_tokenizer(self):
        return self.tokenizer

    def generate(self, messages, stop_sequences=None, grammar=None, **kwargs):
        """
        Gera resposta tratando erros de tipo (List vs String) que o smolagents pode enviar.
        """
        # 1. Limpeza de Argumentos
        clean_kwargs = {
            k: v for k, v in kwargs.items() 
            if k not in ['pipeline', 'grammar', 'stop_sequences', 'adapter_id']
        }

        # 2. Conversão e Sanitização de Mensagens
        formatted_messages = []
        for msg in messages:
            # Identifica Role e Content
            if isinstance(msg, dict):
                role = msg.get('role', 'user')
                content = msg.get('content', '')
            else:
                role = getattr(msg, 'role', 'user')
                content = getattr(msg, 'content', str(msg))

            # --- CORREÇÃO DO ERRO DE CONCATENAÇÃO ---
            # Se o conteúdo for uma lista (ex: uso de ferramenta), converte para string
            if isinstance(content, list):
                content_str = "\n".join([str(item) for item in content])
            else:
                content_str = str(content)
            
            formatted_messages.append({"role": role, "content": content_str})

        # 3. Aplica o template de chat
        try:
            prompt = self.tokenizer.apply_chat_template(
                formatted_messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        except Exception as e:
            # Fallback de emergência se o template falhar
            prompt = str(formatted_messages)

        # 4. Geração
        try:
            outputs = self.pipeline(
                prompt,
                max_new_tokens=2048,
                do_sample=True,
                temperature=0.1,
                top_p=0.95,
                **clean_kwargs
            )
            
            response_text = outputs[0]["generated_text"]
            
            # Retorna objeto ChatMessage
            return ChatMessage(role="assistant", content=response_text)
            
        except Exception as e:
            print(f"Erro na geração: {e}")
            return ChatMessage(role="assistant", content=f"Error: {str(e)}")

# ==============================================================================
#  2. CARREGAMENTO (SETUP)
# ==============================================================================
def get_or_create_engine():
    global GLOBAL_ENGINE
    if GLOBAL_ENGINE is not None:
        return GLOBAL_ENGINE

    print("--- CARREGANDO QWEN 2.5 (4-BIT) ---")
    
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False 
    )

    GLOBAL_ENGINE = LocalQwenEngine(pipeline_obj=text_pipeline)
    print("--- ENGINE PRONTA ---")
    return GLOBAL_ENGINE

# ==============================================================================
#  3. FERRAMENTAS E AGENTES
# ==============================================================================

class PDFSearchTool(Tool):
    name = "search_pdf"
    description = "Search for specific information within the uploaded PDF document."
    inputs = {"query": {"type": "string", "description": "Keywords to search."}}
    output_type = "string"

    def forward(self, query: str) -> str:
        global GLOBAL_RETRIEVER
        if GLOBAL_RETRIEVER is None: return "Error: No PDF uploaded."
        try:
            docs = GLOBAL_RETRIEVER.invoke(query)
            if not docs: return "No info found."
            return "\n\n".join([f"--- Content ---\n{doc.page_content}" for doc in docs])
        except Exception as e: return f"Error: {str(e)}"

class AgentAsTool(Tool):
    def __init__(self, agent, name, description):
        self.agent = agent
        self.name = name
        self.description = description
        self.inputs = {
            "task": {"type": "string", "description": "The task for the agent."}
        }
        self.output_type = "string"
        super().__init__()

    def forward(self, task: str) -> str:
        try:
            return str(self.agent.run(task))
        except Exception as e:
            return f"Error: {str(e)}"

def get_manager_agent():
    engine = get_or_create_engine()

    # 1. Pesquisador
    researcher = CodeAgent(
        tools=[PDFSearchTool()],
        model=engine,
        add_base_tools=False,
        name="pdf_researcher",
        description="Reads PDFs."
    )

    # 2. Ferramenta Pesquisador
    researcher_tool = AgentAsTool(
        agent=researcher,
        name="ask_researcher",
        description="Ask the Researcher to find info in the PDF."
    )

    # 3. Gerente
    manager = CodeAgent(
        tools=[researcher_tool],
        model=engine,
        add_base_tools=False
    )
    return manager

# ==============================================================================
#  4. RAG E CHAT
# ==============================================================================
@spaces.GPU
def build_vector_store(pdf_path):
    global GLOBAL_RETRIEVER
    get_or_create_engine() 
    
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
            collection_name="pdf_store", 
            persist_directory=None
        )
        GLOBAL_RETRIEVER = vectorstore.as_retriever(search_kwargs={"k": 3})
        return True
    except Exception as e:
        print(f"Erro: {e}")
        return False

@spaces.GPU(duration=120)
def chat_function(message, history):
    if not message: return history
    if history is None: history = []
    
    history.append([message, "🤖 Gerente processando..."])
    yield history

    try:
        manager = get_manager_agent()
        
        system_prompt = """
        You are a Manager.
        If the user asks about the PDF, use 'ask_researcher'.
        If the user greets you, answer directly.
        """
        
        response = manager.run(f"{system_prompt}\nUser: {message}")
        history[-1] = [message, str(response)]
        
    except Exception as e:
        history[-1] = [message, f"Erro: {str(e)}"]
        print(traceback.format_exc())
    
    yield history

# ==============================================================================
#  INTERFACE
# ==============================================================================
def process_pdf(file_obj):
    if not file_obj: return "Sem arquivo."
    if build_vector_store(file_obj.name if hasattr(file_obj, 'name') else file_obj):
        return "PDF Pronto! Agentes Ativos."
    return "Erro ao indexar."

with gr.Blocks(title="Multi-Agent List Fix") as demo:
    gr.Markdown("# 🤖 Multi-Agent RAG (Qwen 7B Local)")
    with gr.Row():
        upl = gr.File(label="Upload PDF")
        st = gr.Markdown("...")
    chat = gr.Chatbot(height=600)
    msg = gr.Textbox(label="Pergunta")
    upl.change(process_pdf, upl, st)
    msg.submit(chat_function, [msg, chat], [chat])

if __name__ == "__main__":
    demo.launch()
