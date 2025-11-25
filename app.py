import gradio as gr
import os
import time
import gc
import spaces
from crewai import Agent, Crew, Process, Task
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from langchain_community.llms import HuggingFacePipeline

# Seus imports de ferramentas
from src.agentic_rag.tools.custom_tool import FireCrawlWebSearchTool
from src.agentic_rag.tools.custom_tool import DocumentSearchTool

# Variáveis globais para armazenar o modelo carregado (Cache na RAM/VRAM)
global_model = None
global_tokenizer = None

# ==============================================================================
#                           Carregamento do Modelo (Lazy Loading)
# ==============================================================================
def initialize_model():
    """
    Carrega o modelo uma única vez. As execuções subsequentes usarão o cache global.
    """
    global global_model, global_tokenizer
    
    if global_model is not None and global_tokenizer is not None:
        return global_model, global_tokenizer

    # Modelo leve e rápido recomendado para testes
    model_name = "microsoft/Phi-3.5-mini-instruct"
    print(f"Iniciando carregamento do modelo: {model_name}...")

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
        print(f"Modelo {model_name} carregado com sucesso!")
    except Exception as e:
        print(f"Erro crítico ao carregar modelo: {e}")
        raise e
        
    return global_model, global_tokenizer

# ===========================
#   Configurações LLM
# ===========================
def load_llm():
    # Pega do cache global
    model, tokenizer = initialize_model()

    # Pipeline de geração
    text_generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        # return_full_text=False é importante para evitar repetir o prompt na resposta
        return_full_text=False 
    )

    hf_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    return hf_llm

# ===========================
#   Definição de Agentes e Tarefas
# ===========================

# Esta função roda DENTRO da chat_function, que já tem o decorador @spaces.GPU
# Portanto, não precisa do decorador aqui se for chamada internamente.
def create_agents_and_tasks(pdf_path_str, user_query):
    """
    Cria a Crew fresca para a execução atual.
    """
    
    web_search_tool = FireCrawlWebSearchTool()
    tools_list = [web_search_tool]

    # Instancia a ferramenta de PDF se houver um arquivo
    if pdf_path_str:
        print(f"Indexando PDF para consulta: {pdf_path_str}")
        try:
            # Recria a tool para garantir conexão fresca com a DB vetorial na GPU atual
            pdf_tool = DocumentSearchTool(file_path=pdf_path_str)
            tools_list.append(pdf_tool)
        except Exception as e:
            print(f"Aviso: Falha ao criar PDF Tool: {e}")

    # Carrega o LLM (rápido, pois vem do cache global)
    llm_instance = load_llm()

    # --- Agentes ---
    retriever_agent = Agent(
        role="Information Retriever",
        goal=f"Find relevant information for: {user_query}",
        backstory="You are an expert researcher.",
        verbose=True,
        tools=tools_list,
        llm=llm_instance
    )

    response_agent = Agent(
        role="Helpful Assistant",
        goal="Answer the user query based on retrieved info.",
        backstory="You are a helpful assistant.",
        verbose=True,
        llm=llm_instance
    )

    # --- Tarefas ---
    task1 = Task(
        description=f"Search for information about: {user_query}. Use the PDF tool if available.",
        expected_output="Key findings related to the query.",
        agent=retriever_agent
    )

    task2 = Task(
        description=f"Synthesize the answer for: {user_query} based on the findings.",
        expected_output="A clear text answer.",
        agent=response_agent
    )

    crew = Crew(
        agents=[retriever_agent, response_agent],
        tasks=[task1, task2],
        process=Process.sequential,
        verbose=True
    )
    return crew

# ===========================
#   Funções de Lógica do Gradio
# ===========================

def process_pdf(file_obj):
    """
    Apenas pega o caminho do arquivo.
    """
    if not file_obj:
        return None, "Nenhum arquivo enviado."

    try:
        # Pega o caminho
        file_path = file_obj.name if hasattr(file_obj, 'name') else file_obj
        return file_path, f"PDF carregado! Pronto para perguntas."
    except Exception as e:
        return None, f"Erro no upload: {str(e)}"

@spaces.GPU(duration=120)
def chat_function(message, history, pdf_path_state):
    """
    Função principal. Recria a crew a cada execução para evitar Stale State.
    """
    if not message:
        return history

    # 1. Cria a Crew Fresca (passando o caminho do PDF e a query)
    try:
        crew = create_agents_and_tasks(pdf_path_state, message)
    except Exception as e:
        history.append((message, f"Erro ao criar agentes: {str(e)}"))
        return history

    # 2. Executa
    inputs = {"query": message}
    
    # Placeholder no chat
    history.append((message, "Thinking..."))
    yield history

    try:
        result_obj = crew.kickoff(inputs=inputs)
        final_response = result_obj.raw
    except Exception as e:
        final_response = f"Erro na execução da Crew: {str(e)}"

    # 3. Atualiza o chat com a resposta final
    history[-1] = (message, final_response)
    yield history


# ===========================
#   Interface Gradio
# ===========================

with gr.Blocks(title="Agentic RAG com CrewAI") as demo:

    # Apenas o caminho do PDF é estado persistente. A Crew é efêmera.
    pdf_path_state = gr.State(None)

    gr.Markdown("# Agentic RAG powered by CrewAI")

    with gr.Row():
        with gr.Column(scale=1):
            file_upload = gr.File(label="Upload PDF", file_types=[".pdf"])
            upload_status = gr.Markdown("Aguardando upload...")
            clear_btn = gr.Button("Limpar Chat")

        with gr.Column(scale=4):
            chatbot = gr.Chatbot(label="Histórico", height=600)
            msg_input = gr.Textbox(label="Pergunta", placeholder="Digite aqui...")

    # Evento de Upload
    file_upload.change(
        fn=process_pdf,
        inputs=[file_upload],
        outputs=[pdf_path_state, upload_status]
    )

    # Evento de Chat
    msg_input.submit(
        fn=chat_function,
        inputs=[msg_input, chatbot, pdf_path_state], # Removemos crew_state daqui
        outputs=[chatbot]                            # Removemos crew_state daqui
    ).then(
        fn=lambda: "", outputs=[msg_input]
    )

    def reset_history():
        return []

    clear_btn.click(
        fn=reset_history,
        outputs=[chatbot]
    )

if __name__ == "__main__":
    demo.launch()
