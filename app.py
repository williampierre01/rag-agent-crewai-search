import gradio as gr
import os
import spaces
from typing import Any, List, Optional
from crewai import Agent, Crew, Process, Task
from crewai.tools import BaseTool

# LangChain & Pydantic Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from pydantic import PrivateAttr # <--- IMPORTANTE: Correção do erro de validação

# Transformers Imports
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# ==============================================================================
#  CONFIGURAÇÃO ANTI-ERRO
# ==============================================================================
os.environ["OPENAI_API_KEY"] = "sk-fake-key-bypass"
os.environ["OTEL_SDK_DISABLED"] = "true"

# ==============================================================================
#  GLOBAL STATE
# ==============================================================================
VECTOR_DB_RETRIEVER = None

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
        global VECTOR_DB_RETRIEVER
        if VECTOR_DB_RETRIEVER is None: return "Error: No PDF loaded."
        try:
            docs = VECTOR_DB_RETRIEVER.invoke(query)
            return "\n".join([d.page_content for d in docs])
        except: return "No info found."

# ==============================================================================
#  3. LLM CUSTOMIZADO (CORRIGIDO COM PrivateAttr)
# ==============================================================================
class LocalPhi3(LLM):
    model_name: str = "microsoft/Phi-3.5-mini-instruct"
    
    # Declaramos como atributos privados para o Pydantic não tentar validar/serializar
    _pipeline: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()

    def __init__(self, pipeline, tokenizer, **kwargs):
        super().__init__(**kwargs)
        self._pipeline = pipeline
        self._tokenizer = tokenizer

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        try:
            # Formatação Manual (Chat Template simplificado)
            # Phi-3 espera <|user|> ... <|end|>
            formatted_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>"
            
            response = self._pipeline(
                formatted_prompt, 
                max_new_tokens=1024, 
                return_full_text=False, 
                do_sample=True,
                temperature=0.1
            )
            return response[0]['generated_text']
        except Exception as e:
            # Se der erro aqui, imprimimos para ver no log em vez de deixar o CrewAI tentar LiteLLM
            print(f"[ERRO LLM] Falha na geração: {e}")
            return f"Error generating response: {str(e)}"

    @property
    def _llm_type(self) -> str:
        return "custom_local_phi3"

# Carregador do Modelo
global_llm_instance = None

def load_llm():
    global global_llm_instance
    if global_llm_instance: return global_llm_instance

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

    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )

    # Instanciamos passando os objetos complexos
    global_llm_instance = LocalPhi3(pipeline=text_pipeline, tokenizer=tokenizer)
    
    return global_llm_instance

# ==============================================================================
#  4. AGENTES (Setup Linear)
# ==============================================================================
def create_agents_and_tasks(user_query):
    
    llm = load_llm()
    tools = [PDFRagTool()] if VECTOR_DB_RETRIEVER else []

    # Agente 1: Pesquisa
    researcher = Agent(
        role="Researcher",
        goal="Find facts in the PDF.",
        backstory="Analytical researcher.",
        tools=tools,
        llm=llm,
        verbose=True,
        allow_delegation=False,
        cache=False
    )

    # Agente 2: Resposta
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
        description=f"Search PDF for: '{user_query}'. Return raw facts found.",
        expected_output="List of facts.",
        agent=researcher
    )

    task2 = Task(
        description=f"Answer '{user_query}' based on the facts.",
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
        success = build_vector_store(path)
        if success:
            return "PDF Indexado com Sucesso!"
        else:
            return "Erro ao processar PDF (Verifique Logs)"
    except Exception as e: return f"Erro: {e}"

@spaces.GPU(duration=120)
def chat_function(message, history):
    if not message: return history
    if history is None: history = []
    
    history.append([message, "🤖 Processando (Local)..."])
    yield history

    try:
        crew = create_agents_and_tasks(message)
        result = crew.kickoff()
        history[-1] = [message, str(result.raw)]
    except Exception as e:
        error_msg = f"Erro no Chat: {str(e)}"
        print(error_msg)
        history[-1] = [message, error_msg]
    
    yield history

with gr.Blocks(title="RAG Local Final") as demo:
    gr.Markdown("# 🛡️ RAG Local (Pydantic Fix)")
    with gr.Row():
        upl = gr.File(label="PDF")
        st = gr.Markdown("...")
    chat = gr.Chatbot(height=550)
    msg = gr.Textbox(label="Pergunta")
    
    upl.change(process_pdf, upl, st)
    msg.submit(chat_function, [msg, chat], [chat])

if __name__ == "__main__":
    demo.launch()
