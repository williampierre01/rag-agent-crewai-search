import gradio as gr
import os
import time
import spaces
from crewai import Agent, Crew, Process, Task
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from langchain_community.llms import HuggingFacePipeline

# Seus imports de ferramentas
from src.agentic_rag.tools.custom_tool import FireCrawlWebSearchTool
from src.agentic_rag.tools.custom_tool import DocumentSearchTool

os.environ["OPENAI_API_KEY"] = "NA"

# ==============================================================================
#                           Configuração Global e Cache
# ==============================================================================
global_model = None
global_tokenizer = None

def initialize_model():
    """
    Carrega o modelo Phi-3.5 na memória global para ser reutilizado.
    """
    global global_model, global_tokenizer
    
    if global_model is not None and global_tokenizer is not None:
        return global_model, global_tokenizer

    model_name = "microsoft/Phi-3.5-mini-instruct"
    print(f"[SYSTEM] Carregando modelo LLM: {model_name}...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=bnb_config,
        )

        global_tokenizer = tokenizer
        global_model = model
        print(f"[SYSTEM] Modelo carregado com sucesso!")
    except Exception as e:
        print(f"[ERRO] Falha ao carregar LLM: {e}")
        raise e
        
    return global_model, global_tokenizer

def load_llm():
    """Cria o pipeline do LangChain para o CrewAI."""
    model, tokenizer = initialize_model()

    text_generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.1, # Temperatura baixa para ser mais factual
        top_p=0.95,
        return_full_text=False
    )

    hf_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    return hf_llm

# ==============================================================================
#                           Lógica do CrewAI
# ==============================================================================

def create_agents_and_tasks(pdf_path_str, user_query):
    """
    Cria a Crew. Configura o PDF Tool para NÃO usar OpenAI.
    """
    print(f"[CREW] Iniciando setup para query: {user_query}")
    
    # 1. Configurar FireCrawl (Busca Web)
    # Ele buscará automaticamente a chave em os.environ["FIRECRAWL_API_KEY"]
    try:
        web_search_tool = FireCrawlWebSearchTool()
        tools_list = [web_search_tool]
    except Exception as e:
        print(f"[AVISO] Erro ao carregar FireCrawl (verifique a API Key): {e}")
        tools_list = []

    # 2. Configurar PDF Tool (RAG Local) - O PULO DO GATO
    if pdf_path_str:
        print(f"[CREW] Configurando PDF Tool com Embeddings HuggingFace...")
        try:
            # Esta configuração sobrescreve o padrão da OpenAI
            pdf_tool = DocumentSearchTool(
                file_path=pdf_path_str,
                config={
                    "llm": {
                        "provider": "huggingface",
                        "config": {
                            "model": "microsoft/Phi-3.5-mini-instruct",
                        },
                    },
                    "embedder": {
                        "provider": "huggingface",
                        "config": {
                            "model": "sentence-transformers/all-MiniLM-L6-v2",
                        },
                    },
                }
            )
            tools_list.append(pdf_tool)
        except Exception as e:
            print(f"[ERRO] Falha ao criar PDF Tool: {e}")

    # 3. Carregar LLM
    llm_instance = load_llm()

    # 4. Agentes
    retriever_agent = Agent(
        role="Researcher",
        goal=f"Search for information about: {user_query}",
        backstory="You are an expert researcher. You verify information from the PDF first, then the Web.",
        verbose=True,
        tools=tools_list,
        llm=llm_instance
    )

    response_agent = Agent(
        role="Assistant",
        goal="Answer the user query clearly.",
        backstory="You provide clear, concise answers based on the research provided.",
        verbose=True,
        llm=llm_instance
    )

    # 5. Tarefas
    task1 = Task(
        description=f"Find information to answer: '{user_query}'. Use the PDF tool first. If not enough, use the Web Search.",
        expected_output="A summary of the findings.",
        agent=retriever_agent
    )

    task2 = Task(
        description=f"Based on the findings, write a final answer for: '{user_query}'",
        expected_output="The final text answer.",
        agent=response_agent
    )

    crew = Crew(
        agents=[retriever_agent, response_agent],
        tasks=[task1, task2],
        process=Process.sequential,
        verbose=True
    )
    return crew

# ==============================================================================
#                           Funções do Gradio
# ==============================================================================

def process_pdf(file_obj):
    """Salva apenas o caminho do arquivo."""
    if not file_obj:
        return None, "Nenhum arquivo."
    try:
        file_path = file_obj.name if hasattr(file_obj, 'name') else file_obj
        return file_path, f"PDF carregado: {os.path.basename(file_path)}"
    except Exception as e:
        return None, f"Erro: {str(e)}"

@spaces.GPU(duration=120)
def chat_function(message, history, pdf_path_state):
    """
    Função principal do Chat.
    """
    if not message:
        return history

    if history is None:
        history = []

    # Feedback visual imediato
    history.append([message, "⏳ Pesquisando no PDF e na Web... Aguarde."])
    yield history

    try:
        # Verificar se a chave do FireCrawl existe (apenas aviso no log)
        if "FIRECRAWL_API_KEY" not in os.environ:
            print("[ALERTA] FIRECRAWL_API_KEY não encontrada nas variáveis de ambiente!")

        # 1. Cria a Crew
        crew = create_agents_and_tasks(pdf_path_state, message)
        
        # 2. Executa
        inputs = {"query": message}
        result_obj = crew.kickoff(inputs=inputs)
        final_response = result_obj.raw

        # 3. Atualiza resposta
        history[-1] = [message, final_response]
        yield history

    except Exception as e:
        error_msg = f"❌ Erro: {str(e)}"
        print(f"[ERRO FATAL] {error_msg}")
        history[-1] = [message, error_msg]
        yield history

# ==============================================================================
#                           Interface UI
# ==============================================================================

with gr.Blocks(title="Agentic RAG (Local + FireCrawl)") as demo:

    pdf_path_state = gr.State(None)

    gr.Markdown("# 🤖 Agentic RAG: PDF + Web Search")
    gr.Markdown("Este agente usa **Phi-3.5** (Local), **Embeddings Gratuitos** e **FireCrawl** (Web).")

    with gr.Row():
        with gr.Column(scale=1):
            file_upload = gr.File(label="Upload PDF", file_types=[".pdf"])
            upload_status = gr.Markdown("Status: Aguardando...")
            clear_btn = gr.Button("Limpar")

        with gr.Column(scale=4):
            # CORREÇÃO: type="messages" REMOVIDO para aceitar formato de lista
            chatbot = gr.Chatbot(label="Chat", height=600) 
            msg_input = gr.Textbox(label="Pergunta", placeholder="Digite e dê Enter...")

    file_upload.change(
        fn=process_pdf,
        inputs=[file_upload],
        outputs=[pdf_path_state, upload_status]
    )

    msg_input.submit(
        fn=chat_function,
        inputs=[msg_input, chatbot, pdf_path_state],
        outputs=[chatbot]
    ).then(
        fn=lambda: "", outputs=[msg_input]
    )

    def reset_chat():
        return []

    clear_btn.click(fn=reset_chat, outputs=[chatbot])

if __name__ == "__main__":
    demo.launch()
