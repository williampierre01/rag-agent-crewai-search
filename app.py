import os

# ==============================================================================
#  CONFIGURAÇÕES DE AMBIENTE (DEVEM FICAR NO TOPO ABSOLUTO)
# ==============================================================================
# Definimos isso ANTES de importar qualquer biblioteca para garantir que
# o CrewAI/LiteLLM carreguem essas configurações ao iniciar.

# 1. Chave Falsa (mas com formato que engana validadores regex)
os.environ["OPENAI_API_KEY"] = "sk-proj-fake-key-for-local-execution-12345"

# 2. Redirecionamento para "Buraco Negro" Local
# Dizemos que a API da OpenAI está no próprio container (onde não tem nada).
# Assim, qualquer tentativa de conexão morre instantaneamente sem sair para a internet (evitando erro 401).
os.environ["OPENAI_API_BASE"] = "http://127.0.0.1:11434/v1"

# 3. Desativar Monitoramento e Telemetria (Evita conexões externas)
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["SCARF_NO_ANALYTICS"] = "true"

# ==============================================================================
#  IMPORTS (SÓ AGORA IMPORTAMOS O RESTO)
# ==============================================================================
import gradio as gr
import spaces
import traceback
import gc
from typing import Any, List, Optional
from crewai import Agent, Crew, Process, Task
from crewai.tools import BaseTool

# LangChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from pydantic import PrivateAttr

# Transformers
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# ==============================================================================
#  GLOBAL STATE
# ==============================================================================
GLOBAL_VECTOR_DB = None
GLOBAL_PIPELINE = None 

# ==============================================================================
#  1. BANCO VETORIAL (RAG)
# ==============================================================================
@spaces.GPU
def build_vector_store(pdf_path):
    global GLOBAL_VECTOR_DB
    print(f"[RAG] Indexando: {pdf_path}")
    
    try:
        torch.cuda.empty_cache()
        gc.collect()

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        # Chunks médios para balancear contexto e memória
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)

        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embedding_model,
            collection_name="pdf_rag_store",
            persist_directory=None
        )
        
        GLOBAL_VECTOR_DB = vectorstore.as_retriever(search_kwargs={"k": 3})
        return True
    except Exception as e:
        print(f"[ERRO RAG] {e}")
        traceback.print_exc()
        return False

# ==============================================================================
#  2. FERRAMENTA
# ==============================================================================
class PDFRagTool(BaseTool):
    name: str = "SearchPDF"
    description: str = "Search the PDF content. Input is the search query."

    def _run(self, query: str) -> str:
        if GLOBAL_VECTOR_DB is None: return "Error: No PDF loaded."
        try:
            docs = GLOBAL_VECTOR_DB.invoke(query)
            return "\n".join([d.page_content for d in docs])
        except: return "No info found."

# ==============================================================================
#  3. LLM CUSTOMIZADO (LOCAL QWEN)
# ==============================================================================
class LocalQwen(LLM):
    """
    LLM Local usando Qwen 2.5 7B.
    Finge ser 'gpt-3.5-turbo' para passar nas validações do CrewAI,
    mas roda localmente no Pipeline do HuggingFace.
    """
    model_name: str = "gpt-3.5-turbo" 
    _is_dummy: bool = PrivateAttr(default=True)

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        global GLOBAL_PIPELINE
        
        if GLOBAL_PIPELINE is None:
            return "SYSTEM_ERROR: O modelo Qwen não está carregado na memória."

        try:
            # 1. Limpeza de Argumentos
            # O CrewAI tenta passar callbacks e stops que o pipeline não aceita.
            # Filtramos apenas os parâmetros de geração seguros.
            gen_config = {
                "max_new_tokens": 1024,
                "return_full_text": False,
                "do_sample": True,
                "temperature": 0.1,
                "repetition_penalty": 1.1
            }

            # 2. Formatação do Prompt (ChatML)
            formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            # 3. Execução
            response = GLOBAL_PIPELINE(formatted_prompt, **gen_config)
            
            generated_text = response[0]['generated_text']
            
            # 4. Limpeza da Resposta
            generated_text = generated_text.replace("<|im_end|>", "").strip()
            
            if not generated_text:
                return "Task completed."

            return generated_text

        except Exception as e:
            # Captura erros internos (memória, cuda) e retorna como texto
            print(f"\n!!! ERRO INTERNO NO LLM !!!\n{traceback.format_exc()}")
            return f"Note: I encountered an internal error ({str(e)}). I will proceed with available information."

    @property
    def _llm_type(self) -> str:
        return "custom_local_qwen"

# ==============================================================================
#  CARREGAMENTO GLOBAL
# ==============================================================================
def load_global_model():
    global GLOBAL_PIPELINE
    if GLOBAL_PIPELINE is not None: return

    print("--- CARREGANDO QWEN 7B (4-BIT) ---")
    try:
        model_name = "Qwen/Qwen2.5-7B-Instruct"
        
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

        GLOBAL_PIPELINE = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer
        )
        print("--- QWEN PRONTO ---")
    except Exception as e:
        print(f"ERRO AO CARREGAR MODELO: {e}")
        traceback.print_exc()

# ==============================================================================
#  4. AGENTES
# ==============================================================================
def create_agents_and_tasks(user_query):
    load_global_model()
    
    llm = LocalQwen()
    tools = [PDFRagTool()] if GLOBAL_VECTOR_DB else []

    # Agente
    analyst = Agent(
        role="Analyst",
        goal="Answer the question clearly.",
        backstory="Expert Assistant.",
        tools=tools,
        llm=llm,
        verbose=True,
        allow_delegation=False,
        cache=False,
        max_iter=2
    )

    # Tarefa
    task1 = Task(
        description=f"User query: '{user_query}'. Use the SearchPDF tool if the query is about the document content. If not, answer directly. Be concise.",
        expected_output="The final answer.",
        agent=analyst
    )

    return Crew(
        agents=[analyst],
        tasks=[task1],
        verbose=True,
        process=Process.sequential,
        memory=False, # Crucial para evitar uso de Embeddings OpenAI
        cache=False,
        manager_llm=None,
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
            return "PDF Indexado! Qwen 7B Pronto."
        else:
            return "Erro na indexação."
    except Exception as e: return f"Erro: {e}"

@spaces.GPU(duration=120)
def chat_function(message, history):
    if not message: return history
    if history is None: history = []
    
    history.append([message, "🤖 Processando (Local Qwen)..."])
    yield history

    try:
        crew = create_agents_and_tasks(message)
        result = crew.kickoff()
        history[-1] = [message, str(result.raw)]
    except Exception as e:
        msg = f"Erro: {str(e)}"
        print(f"ERRO FINAL: {traceback.format_exc()}")
        history[-1] = [message, msg]
    
    yield history

with gr.Blocks(title="CrewAI Qwen Space") as demo:
    gr.Markdown("# 🛡️ CrewAI Local no Hugging Face")
    gr.Markdown("Powered by **Qwen 2.5 7B** (ZeroGPU)")
    
    with gr.Row():
        upl = gr.File(label="Upload PDF")
        st = gr.Markdown("...")
    chat = gr.Chatbot(height=550)
    msg = gr.Textbox(label="Pergunta")
    
    upl.change(process_pdf, upl, st)
    msg.submit(chat_function, [msg, chat], [chat])

if __name__ == "__main__":
    demo.launch()
