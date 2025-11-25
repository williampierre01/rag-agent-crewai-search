import gradio as gr
import os
import time
import spaces
from crewai import Agent, Crew, Process, Task
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from langchain_community.llms import HuggingFacePipeline

# Seus imports de ferramentas (certifique-se que o caminho src está correto)
from src.agentic_rag.tools.custom_tool import FireCrawlWebSearchTool
from src.agentic_rag.tools.custom_tool import DocumentSearchTool

# ==============================================================================
#                           Configuração Global e Cache
# ==============================================================================
# Variáveis para armazenar o modelo na RAM/VRAM e não carregar toda vez
global_model = None
global_tokenizer = None

def initialize_model():
    """
    Carrega o modelo apenas se ainda não estiver na memória global.
    """
    global global_model, global_tokenizer
    
    if global_model is not None and global_tokenizer is not None:
        return global_model, global_tokenizer

    # Modelo eficiente para rodar no ZeroGPU
    model_name = "microsoft/Phi-3.5-mini-instruct"
    print(f"[SYSTEM] Iniciando carregamento do modelo: {model_name}...")

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
        print(f"[SYSTEM] Modelo {model_name} carregado com sucesso!")
    except Exception as e:
        print(f"[ERRO] Falha crítica ao carregar modelo: {e}")
        raise e
        
    return global_model, global_tokenizer

def load_llm():
    """Cria o pipeline do LangChain/HuggingFace."""
    model, tokenizer = initialize_model()

    text_generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        return_full_text=False # Importante para não repetir a pergunta
    )

    hf_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    return hf_llm

# ==============================================================================
#                           Lógica do CrewAI
# ==============================================================================

def create_agents_and_tasks(pdf_path_str, user_query):
    """
    Cria os agentes, tarefas e ferramentas para a execução atual.
    """
    print(f"[CREW] Montando equipe para query: {user_query}")
    
    # 1. Ferramentas
    web_search_tool = FireCrawlWebSearchTool()
    tools_list = [web_search_tool]

    if pdf_path_str:
        print(f"[CREW] Adicionando ferramenta de PDF: {pdf_path_str}")
        try:
            # Recria a ferramenta para garantir conexão válida na GPU atual
            pdf_tool = DocumentSearchTool(file_path=pdf_path_str)
            tools_list.append(pdf_tool)
        except Exception as e:
            print(f"[AVISO] Erro ao criar PDF Tool (seguindo sem ela): {e}")

    # 2. LLM
    llm_instance = load_llm()

    # 3. Agentes
    retriever_agent = Agent(
        role="Senior Researcher",
        goal=f"Find specific information to answer: {user_query}",
        backstory="You are an expert at finding information in PDFs and the Web.",
        verbose=True,
        tools=tools_list,
        llm=llm_instance
    )

    response_agent = Agent(
        role="Customer Support Lead",
        goal="Synthesize the found information into a clear answer.",
        backstory="You provide concise and helpful answers based on research.",
        verbose=True,
        llm=llm_instance
    )

    # 4. Tarefas
    task1 = Task(
        description=f"Search for '{user_query}'. Use the PDF tool first if available.",
        expected_output="Key findings regarding the query.",
        agent=retriever_agent
    )

    task2 = Task(
        description=f"Write a final answer for '{user_query}' based on the findings.",
        expected_output="A helpful text response.",
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
    """Salva apenas o caminho do arquivo para evitar erros de serialização."""
    if not file_obj:
        return None, "Nenhum arquivo recebido."

    try:
        file_path = file_obj.name if hasattr(file_obj, 'name') else file_obj
        return file_path, f"PDF carregado: {os.path.basename(file_path)}"
    except Exception as e:
        return None, f"Erro no upload: {str(e)}"

@spaces.GPU(duration=120)
def chat_function(message, history, pdf_path_state):
    """
    Função principal com tratamento de erros e feedback visual.
    """
    print(f"\n--- NOVA INTERAÇÃO: {message} ---")
    
    if not message:
        return history

    if history is None:
        history = []

    # 1. Feedback Imediato (Usuario vê que algo está rodando)
    history.append([message, "🔍 Processando... Por favor, aguarde."])
    yield history

    try:
        # 2. Criação da Crew
        print("[STEP 1] Criando Agentes...")
        crew = create_agents_and_tasks(pdf_path_state, message)
        
        # 3. Execução
        print("[STEP 2] Iniciando Kickoff...")
        inputs = {"query": message}
        result_obj = crew.kickoff(inputs=inputs)
        final_response = result_obj.raw
        
        print("[STEP 3] Sucesso!")

        # 4. Atualiza resposta final
        history[-1] = [message, final_response]
        yield history

    except Exception as e:
        # Se der erro, mostra no chat para você saber o que foi
        error_msg = f"❌ Ocorreu um erro: {str(e)}"
        print(f"[ERRO FATAL] {error_msg}")
        history[-1] = [message, error_msg]
        yield history

# ==============================================================================
#                           Interface UI
# ==============================================================================

with gr.Blocks(title="Agentic RAG com CrewAI") as demo:

    # Estado apenas para o caminho (String)
    pdf_path_state = gr.State(None)

    gr.Markdown("# 🤖 Agentic RAG powered by CrewAI")
    gr.Markdown("Faça upload de um PDF e faça perguntas. O sistema usará Agentes para pesquisar no PDF e na Web.")

    with gr.Row():
        with gr.Column(scale=1):
            file_upload = gr.File(label="Upload PDF", file_types=[".pdf"])
            upload_status = gr.Markdown("Status: Aguardando arquivo...")
            clear_btn = gr.Button("🗑️ Limpar Conversa")

        with gr.Column(scale=4):
            chatbot = gr.Chatbot(label="Chat", height=600, type="messages")
            # Nota: type="messages" é o novo padrão, mas se der erro visual, remova esse parametro.
            
            msg_input = gr.Textbox(label="Sua Pergunta", placeholder="Digite aqui e pressione Enter...")

    # Eventos
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
        fn=lambda: "", outputs=[msg_input] # Limpa caixa de texto
    )

    def reset_chat():
        return []

    clear_btn.click(
        fn=reset_chat,
        outputs=[chatbot]
    )

if __name__ == "__main__":
    demo.launch()
