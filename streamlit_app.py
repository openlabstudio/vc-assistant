import os, base64
from pathlib import Path
import hmac
import tempfile
import pandas as pd
import uuid

import streamlit as st

from langchain_community.vectorstores import AstraDB
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import AstraDBChatMessageHistory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, WebBaseLoader
from langchain.schema import HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap
from langchain.schema import StrOutputParser

from langchain.callbacks.base import BaseCallbackHandler

import openai

# --- CONFIGURACIÓN GLOBAL ---
ASTRA_DB_COLLECTION_NAME = "vc_assistant"
ADMIN_USERS = ["openlab_admin"] 


print("Streamlit App Started")
st.set_page_config(page_title='AI VC Assistant', page_icon='🚀', layout="wide")

# --- INICIALIZACIÓN DE SESSION STATE ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "debug_messages" not in st.session_state:
    st.session_state.debug_messages = []


# --- CLASES Y FUNCIONES ---
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text
    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text + "▌")

def check_password():
    def login_form():
        with st.form("credentials"):
            st.text_input('Username', key='username')
            st.text_input('Password', type='password', key='password')
            st.form_submit_button('Login', on_click=password_entered)

    def password_entered():
        if 'passwords' in st.secrets and st.session_state.get('username') in st.secrets['passwords'] and hmac.compare_digest(st.session_state.get('password', ''), st.secrets.passwords[st.session_state.get('username', '')]):
            st.session_state['password_correct'] = True
            st.session_state.user = st.session_state['username']
            if 'password' in st.session_state: del st.session_state['password']
        else:
            st.session_state['password_correct'] = False

    if st.session_state.get('password_correct', False):
        return True
    login_form()
    if "password_correct" in st.session_state and not st.session_state.password_correct:
        st.error('😕 Usuario desconocido o contraseña incorrecta')
    return False

def logout():
    keys_to_delete = list(st.session_state.keys())
    for key in keys_to_delete: del st.session_state[key]
    st.cache_resource.clear()
    st.cache_data.clear()
    st.rerun()

def vectorize_text(uploaded_files, vectorstore, lang_dict):
    if not vectorstore:
        st.error(lang_dict.get('vectorstore_not_ready_admin', "Vectorstore not ready for upload."))
        return
    with st.spinner("Processing files... This may take a moment."):
        for uploaded_file in uploaded_files:
            if uploaded_file is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                st.session_state.debug_messages.append(f"Admin: Processing file {uploaded_file.name}")
                try:
                    docs = []
                    if uploaded_file.name.endswith('.pdf'):
                        loader = PyPDFLoader(tmp_file_path)
                        docs = loader.load_and_split()
                    elif uploaded_file.name.endswith('.csv'):
                        loader = CSVLoader(tmp_file_path, encoding='utf-8')
                        docs = loader.load()
                    elif uploaded_file.name.endswith('.txt'):
                        from langchain.schema import Document
                        with open(tmp_file_path, 'r', encoding='utf-8') as f_txt:
                            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                            docs = text_splitter.create_documents([f_txt.read()], metadatas=[{"source": uploaded_file.name}])
                    else:
                        st.warning(f"Unsupported file type: {uploaded_file.name}")
                        continue
                    
                    if docs:
                        vectorstore.add_documents(docs)
                        st.info(f"✅ {uploaded_file.name} processed ({len(docs)} segments).")
                except Exception as e:
                    st.error(f"Error processing file {uploaded_file.name}: {e}")
                finally:
                    os.remove(tmp_file_path)

def vectorize_url(urls, vectorstore, lang_dict):
    if not vectorstore:
        st.error(lang_dict.get('vectorstore_not_ready_admin', "Vectorstore not ready for URL load."))
        return
    with st.spinner("Processing URLs..."):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        for url in urls:
            url = url.strip()
            if not url: continue
            try:
                st.session_state.debug_messages.append(f"Admin: Loading from URL: {url}")
                loader = WebBaseLoader(url)
                docs = loader.load()
                pages = text_splitter.split_documents(docs)
                vectorstore.add_documents(pages)
                st.info(f"✅ URL processed: {url}")
            except Exception as e:
                st.error(f"Error loading from URL {url}: {e}")

def get_prompt(type, custom_prompt, language):
    base_template = f"""Use the following context to answer the question:
{{context}}

Use the following chat history to answer the question:
{{chat_history}}

Question:
{{question}}

Answer in {language}:"""
    
    if type == 'Extended results':
        template = f"""You're a helpful AI assistant tasked to answer the user's questions.
You're friendly and you answer extensively with multiple sentences. You prefer to use bulletpoints to summarize.
If you don't know the answer, just say 'I do not know the answer'.
{base_template}"""
    elif type == 'Short results':
        template = f"""You're a helpful AI assistant tasked to answer the user's questions.
You answer in an exceptionally brief way.
If you don't know the answer, just say 'I do not know the answer'.
{base_template}"""
    else: # 'Custom'
        template = custom_prompt if custom_prompt else base_template
    
    return ChatPromptTemplate.from_messages([("system", template)])

# ---- FUNCIÓN AÑADIDA ----
def load_retriever(vectorstore, top_k_vectorstore):
    print(f"""load_retriever with top_k_vectorstore='{top_k_vectorstore}'""")
    # Get the Retriever from the Vectorstore
    return vectorstore.as_retriever(
        search_kwargs={"k": top_k_vectorstore}
    )
# -------------------------

def generate_queries(model, language):
    prompt = f"""You are a helpful assistant that generates multiple search queries based on a single input query in language {language}.
Generate multiple search queries related to: {{original_query}}
OUTPUT (4 queries):"""
    return ChatPromptTemplate.from_messages([("system", prompt)]) | model | StrOutputParser() | (lambda x: x.split("\n"))

def reciprocal_rank_fusion(results: list[list], k=60):
    from langchain.load import dumps, loads
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            try:
                doc_str = dumps(doc)
                if doc_str not in fused_scores: fused_scores[doc_str] = 0
                fused_scores[doc_str] += 1 / (rank + k)
            except: continue
    return [(loads(doc), score) for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)]


# --- Funciones de Carga de Recursos Cacheadas ---
@st.cache_data()
def load_localization(locale):
    try:
        df = pd.read_csv("./customizations/localization.csv", encoding='utf-8')
        df_lang = df.query(f"locale == '{locale}'")
        if df_lang.empty: df_lang = df.query("locale == 'en_US'")
        return pd.Series(df_lang.value.values, index=df_lang.key).to_dict()
    except Exception: return {"assistant_welcome": "Welcome (localization file not found)."}

@st.cache_data()
def load_rails(username):
    try:
        df = pd.read_csv("./customizations/rails.csv", encoding='utf-8')
        df_user = df.query(f"username == '{username}'")
        return pd.Series(df_user.value.values, index=df_user.key).to_dict()
    except Exception: return {}

@st.cache_resource(show_spinner="Loading embeddings...")
def load_embedding_rc():
    return OpenAIEmbeddings(openai_api_key=st.secrets.get("OPENAI_API_KEY"))

@st.cache_resource(show_spinner="Loading vector store...")
def load_vectorstore_rc(username, _embedding): # Underscore to prevent hashing
    return AstraDB(
        embedding=_embedding,
        collection_name=ASTRA_DB_COLLECTION_NAME,
        token=st.secrets["ASTRA_TOKEN"],
        api_endpoint=st.secrets.get("ASTRA_ENDPOINT", os.environ.get("ASTRA_ENDPOINT")),
        namespace=st.secrets.get("ASTRA_KEYSPACE")
    )

@st.cache_resource(show_spinner="Loading chat history...")
def load_chat_history_rc(username, session_id):
    return AstraDBChatMessageHistory(
        session_id=f"{username}_{str(session_id)}",
        api_endpoint=st.secrets.get("ASTRA_ENDPOINT", os.environ.get("ASTRA_ENDPOINT")),
        token=st.secrets["ASTRA_TOKEN"],
        collection_name="historial_chat_asistente"
    )

@st.cache_resource(show_spinner="Loading AI model...")
def load_model_rc():
    return ChatOpenAI(temperature=0.3, model='gpt-4o', streaming=True, verbose=False)

@st.cache_resource()
def load_memory_rc(_chat_history, top_k_history):
    return ConversationBufferWindowMemory(
        chat_memory=_chat_history, return_messages=True, k=top_k_history,
        memory_key="chat_history", input_key="question", output_key='answer')

# --- FLUJO PRINCIPAL DE LA APP ---

# 1. Login
if not check_password():
    st.stop()

# 2. Inicialización Post-Login
username = st.session_state.user
language = st.secrets.get("languages", {}).get(username, "es_ES")
lang_dict = load_localization(language)

# Inicializar recursos
embedding = load_embedding_rc()
vectorstore = load_vectorstore_rc(username, embedding) if embedding else None
chat_history = load_chat_history_rc(username, st.session_state.session_id) if vectorstore else None
model = load_model_rc()

# 3. Configuración de Parámetros (Condicional según el usuario)
if username != "demo":
    # --- SIDEBAR PARA USUARIOS NORMALES / ADMINS ---
    with st.sidebar:
        # Logo, Logout, Rails...
        st.image('./customizations/logo/default.svg', use_column_width="always")
        st.markdown(f"""{lang_dict.get('logout_caption','Logged in as')} :orange[{username}]""")
        if st.button(lang_dict.get('logout_button','Logout')): logout()
        st.divider()

        rails_dict = load_rails(username)
        st.subheader(lang_dict.get('rails_1', "Suggestions"))
        st.caption(lang_dict.get('rails_2', "Try asking:"))
        if rails_dict:
            for i in sorted(rails_dict.keys()): st.markdown(f"{i}. {rails_dict[i]}")
        st.divider()

        # --- Opciones de Chat (Leen los defaults de secrets.toml) ---
        st.subheader(lang_dict.get('options_header', "Chat Options"))

        # Cargar los defaults para el usuario actual desde secrets.toml
        user_defaults = st.secrets.get("DEFAULT_SETTINGS", {}).get(username, {})

        disable_chat_history = st.toggle(lang_dict.get('disable_chat_history', "Disable Chat History"), value=False)
        top_k_history = st.slider(
            lang_dict.get('k_chat_history', "K for Chat History"), 1, 10,
            value=user_defaults.get("TOP_K_HISTORY", 5), # <-- VALOR POR DEFECTO DESDE SECRETS
            disabled=disable_chat_history
        )
        disable_vector_store = st.toggle(lang_dict.get('disable_vector_store', "Disable Vector Store?"), value=False)
        top_k_vectorstore = st.slider(
            lang_dict.get('top_k_vector_store', "Top-K for Vector Store"), 1, 50,
            value=user_defaults.get("TOP_K_VECTORSTORE", 5), # <-- VALOR POR DEFECTO DESDE SECRETS
            disabled=disable_vector_store
        )
        
        rag_strategies = ('Basic Retrieval', 'Maximal Marginal Relevance', 'Fusion')
        default_strategy = user_defaults.get("RAG_STRATEGY", 'Basic Retrieval')
        strategy_idx = rag_strategies.index(default_strategy) if default_strategy in rag_strategies else 0
        strategy = st.selectbox(
            lang_dict.get('rag_strategy', "RAG Strategy"), rag_strategies,
            index=strategy_idx, # <-- VALOR POR DEFECTO DESDE SECRETS
            disabled=disable_vector_store
        )
        
        # Carga de prompt personalizado
        custom_prompt_text_val = ""
        prompt_options = ('Short results', 'Extended results', 'Custom')
        default_prompt_type = user_defaults.get("PROMPT_TYPE", 'Custom')
        prompt_idx = prompt_options.index(default_prompt_type) if default_prompt_type in prompt_options else 2

        try:
            user_prompt_file = Path(f"./customizations/prompt/{username}.txt")
            default_prompt_file = Path("./customizations/prompt/default.txt")
            if user_prompt_file.is_file():
                custom_prompt_text_val = user_prompt_file.read_text(encoding='utf-8')
            elif default_prompt_file.is_file():
                custom_prompt_text_val = default_prompt_file.read_text(encoding='utf-8')
            else:
                # Pega aquí tu prompt avanzado completo como fallback
                custom_prompt_text_val = """### ROL Y PERSONALIDAD ###
Actúa como un Analista Estratégico Senior..."""
        except Exception as e:
            st.warning(f"Could not load prompt file: {e}")
            custom_prompt_text_val = "Error loading prompt file."

        prompt_type = st.selectbox(
            lang_dict.get('system_prompt', "System Prompt"), prompt_options,
            index=prompt_idx # <-- VALOR POR DEFECTO DESDE SECRETS
        )
        custom_prompt = st.text_area(
            lang_dict.get('custom_prompt', "Custom Prompt"),
            value=custom_prompt_text_val, 
            disabled=(prompt_type != 'Custom')
        )
        st.divider()

        # Herramientas de Administración (Solo visibles para admins)
        if username in ADMIN_USERS:
            st.subheader(lang_dict.get('admin_tools_header', "Admin Tools"))
            with st.expander("Upload Files"):
                uploaded_files = st.file_uploader("Upload TXT, PDF, CSV files", type=['txt', 'pdf', 'csv'], accept_multiple_files=True)
                if st.button("Process Files"):
                    if uploaded_files: vectorize_text(uploaded_files, vectorstore, lang_dict)
            with st.expander("Load from URLs"):
                urls_text = st.text_area("Enter URLs (comma-separated)")
                if st.button("Process URLs"):
                    urls = [url.strip() for url in urls_text.split(',') if url.strip()]
                    if urls: vectorize_url(urls, vectorstore, lang_dict)

else: # Si el usuario es 'demo', la sidebar no se muestra y usamos valores por defecto.
    # Inyectar CSS para ocultar la sidebar
    st.markdown("""<style>[data-testid="stSidebar"] {display: none}</style>""", unsafe_allow_html=True)
    
    # Cargar los defaults para el usuario 'demo' desde secrets.toml
    user_defaults = st.secrets.get("DEFAULT_SETTINGS", {}).get(username, {})

    # Definir valores por defecto para que la app no falle
    disable_chat_history = False
    top_k_history = user_defaults.get("TOP_K_HISTORY", 5)
    disable_vector_store = False
    top_k_vectorstore = user_defaults.get("TOP_K_VECTORSTORE", 3)
    strategy = user_defaults.get("RAG_STRATEGY", 'Basic Retrieval')
    prompt_type = user_defaults.get("PROMPT_TYPE", 'Custom')
    
    # Cargar el prompt personalizado para el modo demo
    try:
        custom_prompt = Path(f"./customizations/prompt/{username}.txt").read_text(encoding='utf-8')
    except:
        try:
            custom_prompt = Path("./customizations/prompt/default.txt").read_text(encoding='utf-8')
        except:
            # Pega aquí tu prompt avanzado completo como fallback
            custom_prompt = """### ROL Y PERSONALIDAD ###
Actúa como un Analista Estratégico Senior..."""

# Inicializar memoria (fuera del if/else para que ambos tipos de usuario la tengan)
memory = load_memory_rc(chat_history, top_k_history if not disable_chat_history else 0) if chat_history else None

# --- Interfaz Principal del Chat (Visible para TODOS los usuarios) ---
if 'messages' not in st.session_state:
    st.session_state.messages = [AIMessage(content=lang_dict.get('assistant_welcome', "Welcome!"))]
for message in st.session_state.messages:
    st.chat_message(message.type).markdown(message.content)

if question := st.chat_input(lang_dict.get('assistant_question', "Your question...")):
    st.session_state.messages.append(HumanMessage(content=question))
    with st.chat_message('human'):
        st.markdown(question)

    with st.chat_message('assistant'):
        response_placeholder = st.empty()
        
        if not model or (not disable_vector_store and not vectorstore):
            response_placeholder.markdown("Sorry, the assistant is not fully configured. Please contact the administrator.")
        else:
            relevant_documents = []
            if not disable_vector_store:
                retriever = load_retriever(vectorstore, top_k_vectorstore)
                if retriever:
                    if strategy == 'Basic Retrieval':
                        relevant_documents = retriever.get_relevant_documents(query=question)
                    elif strategy == 'Maximal Marginal Relevance':
                        relevant_documents = vectorstore.max_marginal_relevance_search(query=question, k=top_k_vectorstore)
                    elif strategy == 'Fusion':
                        generate_queries_chain_instance = generate_queries(model, language)
                        if generate_queries_chain_instance:
                            fusion_queries = generate_queries_chain_instance.invoke({"original_query": question})
                            chain_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion
                            fused_results = chain_fusion.invoke({"original_query": question})
                            relevant_documents = [doc_tuple[0] for doc_tuple in fused_results][:top_k_vectorstore]
            
            history = memory.load_memory_variables({}) if memory else {"chat_history": []}
            
            rag_chain_inputs = RunnableMap({
                'context': lambda x: x['context'],
                'chat_history': lambda x: x['chat_history'],
                'question': lambda x: x['question']
            })
            current_prompt_obj = get_prompt(prompt_type, custom_prompt, language)
            chain = rag_chain_inputs | current_prompt_obj | model

            try:
                response = chain.invoke(
                    {'question': question, 'chat_history': history.get('chat_history', []), 'context': relevant_documents}, 
                    config={'callbacks': [StreamHandler(response_placeholder)]}
                )
                final_content = response.content
                if memory: memory.save_context({'question': question}, {'answer': final_content})

                if not disable_vector_store and relevant_documents:
                    final_content += f"\n\n*{lang_dict.get('sources_used','Fuentes:')}*"
                    sources_used = []
                    for doc in relevant_documents:
                        source_name = doc.metadata.get('source', 'Unknown')
                        source_basename = os.path.basename(os.path.normpath(source_name))
                        if source_basename not in sources_used:
                            final_content += f"\n📙 :orange[{source_basename}]"
                            sources_used.append(source_basename)
                
                response_placeholder.markdown(final_content)
                st.session_state.messages.append(AIMessage(content=final_content))
            except Exception as e:
                st.error(f"Error during response generation: {e}")
