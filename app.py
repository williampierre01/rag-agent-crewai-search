import gradio as gr
import os
import spaces
import traceback
from typing import Any, List, Optional
from crewai import Agent, Crew, Process, Task
from crewai.tools import BaseTool

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

# Transformers Imports
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# ==============================================================================
#  CONFIGURAÇÕES DE AMBIENTE
# ==============================================================================
os.environ["OPENAI_API_KEY"] = "sk-fake-key-bypass"
os.environ["OTEL_SDK_DISABLED"] = "true"

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
        
        GLOBAL_VECTOR_DB = vectorstore.as_retriever(search_kwargs={"k": 4})
        return True
    except Exception as e:
        print(f"[ERRO RAG] {e}")
        traceback.print_exc()
        return False

# ==============================================================================
#  2. FERRAMENTA CUSTOMIZADA
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
#  3. LLM CUSTOMIZADO BLINDADO (A SOLUÇÃO)
# ==============================================================================
class UnbreakableLLM(LLM):
    """
    Uma classe LLM que engole qualquer erro e retorna como texto,
    impedindo que o CrewAI tente fazer fallback para OpenAI.
    """
    model_name: str = "microsoft/Phi-3.5-mini-instruct"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        # Acessa a variável global
        global GLOBAL_PIPELINE
        
        if GLOBAL_PIPELINE is None:
            return "SYSTEM_ERROR: O Modelo Global não está carregado na GPU."

        try:
            # 1. Formatação Manual (Simples e Eficaz)
            # Removemos qualquer tentativa complexa de chat templates do LangChain
            formatted_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>"
            
            # 2. Execução
            # Ignoramos **kwargs que o CrewAI manda (como callbacks complexos)
            # Ignoramos 'stop' words pois o HF Pipeline as vezes trava com elas
            response = GLOBAL_PIPELINE(
                formatted_prompt, 
                max_new_tokens=1024, 
                return_full_text=False, 
                do_sample=True,
                temperature=0.1
            )
            
            text = response[0]['generated_text']
            # Limpeza
            return text.replace("<|end|>", "").strip()

        except Exception as e:
            # AQUI ESTÁ O TRUQUE:
            # Imprimimos o erro no terminal (para você ver)
            print("\n!!! ERRO DENTRO DO LLM !!!")
            traceback.print_exc()
            
            # E retornamos uma string normal. 
            # O CrewAI vai achar que essa foi a resposta do modelo e seguirá em frente.
            return f"Desculpe, ocorreu um erro interno na geração: {str(e)}"

    @property
    def _llm_type(self) -> str:
        return "custom_unbreakable"

# Carregamento Global
def load_global_model():
    global GLOBAL_PIPELINE
    if GLOBAL_PIPELINE is not None: return

    print("--- LOAD MODEL ---")
    try:
        model_name = "microsoft/Phi-3.5-mini-instruct"
        
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
        print("--- MODEL READY ---")
    except Exception as e:
        print(f"FATAL ERROR LOADING MODEL: {e}")
        traceback.print_exc()

# ==============================================================================
#  4. AGENTES
# ==============================================================================
def create_agents_and_tasks(user_query):
    load_global_model()
    
    # Usamos nossa classe blindada
    llm = UnbreakableLLM()
    
    tools = [PDFRagTool()] if GLOBAL_VECTOR_DB else []

    researcher = Agent(
        role="Researcher",
        goal="Find facts.",
        backstory="Researcher.",
        tools=tools,
        llm=llm,
        verbose=True,
        allow_delegation=False,
        cache=False,
    )

    writer = Agent(
        role="Writer",
        goal="Summarize.",
        backstory="Writer.",
        tools=[], 
        llm=llm,
        verbose=True,
        allow_delegation=False,
        cache=False
    )

    task1 = Task(
        description=f"Search PDF for '{user_query}' and list facts.",
        expected_output="Facts list.",
        agent=researcher
    )

    task2 = Task(
        description=f"Answer '{user_query}' based on facts.",
        expected_output="Final answer.",
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
    
    history.append([message, "🤖 Iniciando..."])
    yield history

    try:
        crew = create_agents_and_tasks(message)
        result = crew.kickoff()
        history[-1] = [message, str(result.raw)]
    except Exception as e:
        msg = f"Erro Crew: {str(e)}"
        print(msg)
        traceback.print_exc()
        history[-1] = [message, msg]
    
    yield history

with gr.Blocks(title="RAG Final Unbreakable") as demo:
    gr.Markdown("# 🛡️ RAG Local (Unbreakable Version)")
    with gr.Row():
        upl = gr.File(label="PDF")
        st = gr.Markdown("...")
    chat = gr.Chatbot(height=550)
    msg = gr.Textbox(label="Pergunta")
    
    upl.change(process_pdf, upl, st)
    msg.submit(chat_function, [msg, chat], [chat])

if __name__ == "__main__":
    demo.launch()
