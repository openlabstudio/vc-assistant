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

print("Started")
st.set_page_config(page_title='Your Enterprise Sidekick', page_icon='🚀')

# Get a unique session id for memory
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Streaming call back handler for responses
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text + "▌")

#################
### Functions ###
#################

# Close off the app using a password
def check_password():
    """Returns `True` if the user had a correct password."""

    def login_form():
        """Form with widgets to collect user information"""
        with st.form("credentials"):
            st.text_input('Username', key='username')
            st.text_input('Password', type='password', key='password')
            st.form_submit_button('Login', on_click=password_entered)

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if 'passwords' in st.secrets and st.session_state['username'] in st.secrets['passwords'] and hmac.compare_digest(st.session_state['password'], st.secrets.passwords[st.session_state['username']]):
            st.session_state['password_correct'] = True
            st.session_state.user = st.session_state['username']
            if 'password' in st.session_state:
                del st.session_state['password']  # Don't store the password.
        else:
            st.session_state['password_correct'] = False

    if st.session_state.get('password_correct', False):
        return True

    login_form()
    if "password_correct" in st.session_state and not st.session_state.password_correct:
        st.error('😕 Usuario desconocido o contraseña incorrecta')
    return False

def logout():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.cache_resource.clear()
    st.cache_data.clear()
    st.rerun()

# Function for Vectorizing uploaded data into Astra DB
def vectorize_text(uploaded_files, vectorstore, lang_dict):
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getvalue())

                if uploaded_file.name.endswith('.txt'):
                    with open(file_path, 'r', encoding='utf-8') as f_txt:
                        file_content_list = [f_txt.read()]
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
                    texts = text_splitter.create_documents(file_content_list, [{'source': uploaded_file.name}] * len(file_content_list))
                    vectorstore.add_documents(texts)
                    st.info(f"{len(texts)} {lang_dict.get('load_text', 'text segments loaded')}")
                
                elif uploaded_file.name.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
                    pages = text_splitter.split_documents(docs)
                    vectorstore.add_documents(pages)  
                    st.info(f"{len(pages)} {lang_dict.get('load_pdf', 'PDF pages/segments loaded')}")

                elif uploaded_file.name.endswith('.csv'):
                    loader = CSVLoader(file_path, encoding='utf-8')
                    docs = loader.load()
                    vectorstore.add_documents(docs)
                    st.info(f"{len(docs)} {lang_dict.get('load_csv', 'CSV documents/rows loaded')}")

# Load data from URLs
def vectorize_url(urls, vectorstore, lang_dict):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    for url in urls:
        url = url.strip()
        if not url: continue
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()    
            pages = text_splitter.split_documents(docs)
            print (f"Loading from URL: {pages}")
            vectorstore.add_documents(pages)  
            st.info(f"{len(pages)} {lang_dict.get('url_pages_loaded', 'loaded')}")
        except Exception as e:
            st.info(f"{lang_dict.get('url_error', 'An error occurred')}: {e}")

# Define the prompt
def get_prompt(type, custom_prompt, language):
    template = ''
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
If the question states the name of the user, just say 'Thanks, I'll use this information going forward'.
If you don't know the answer, just say 'I do not know the answer'.
{base_template}"""

    elif type == 'Short results':
        template = f"""You're a helpful AI assistant tasked to answer the user's questions.
You answer in an exceptionally brief way.
If the question states the name of the user, just say 'Thanks, I'll use this information going forward'.
If you don't know the answer, just say 'I do not know the answer'.
{base_template}"""

    elif type == 'Custom':
        template = custom_prompt
    
    return ChatPromptTemplate.from_messages([("system", template)])

# Get the OpenAI Chat Model
def load_model():
    print(f"""load_model""")
    return ChatOpenAI(temperature=0.3, model='gpt-4-1106-preview', streaming=True, verbose=True)

# Get the Retriever
def load_retriever(vectorstore, top_k_vectorstore):
    print(f"""load_retriever with top_k_vectorstore='{top_k_vectorstore}'""")
    return vectorstore.as_retriever(search_kwargs={"k": top_k_vectorstore})

@st.cache_resource()
def load_memory(_chat_history, top_k_history):
    print(f"""load_memory with top-k={top_k_history}""")
    return ConversationBufferWindowMemory(
        chat_memory=_chat_history, return_messages=True, k=top_k_history,
        memory_key="chat_history", input_key="question", output_key='answer')

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
            doc_str = dumps(doc)
            if doc_str not in fused_scores: fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)
    reranked_results = [(loads(doc), score) for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)]
    return reranked_results

def describeImage(image_bin, language):
    print ("describeImage")
    image_base64 = base64.b64encode(image_bin).decode()
    client = openai.OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": f"Provide a search text for the main topic of the image writen in {language}"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
        ]}], max_tokens=4096)
    print (f"describeImage result: {response}")
    return response

# Cache localized strings
@st.cache_data()
def load_localization(locale):
    print("load_localization")
    try:
        df = pd.read_csv("./customizations/localization.csv", encoding='utf-8')
        df_lang = df.query(f"locale == '{locale}'")
        if df_lang.empty: # Fallback to en_US if locale not found
            df_lang = df.query("locale == 'en_US'")
        return pd.Series(df_lang.value.values, index=df_lang.key).to_dict()
    except Exception as e:
        print(f"Could not load localization file. Using empty dict. Error: {e}")
        return {} # Return empty dict on error

# Cache localized strings
@st.cache_data()
def load_rails(username):
    print("load_rails")
    try:
        df = pd.read_csv("./customizations/rails.csv", encoding='utf-8')
        df_user = df.query(f"username == '{username}'")
        return pd.Series(df_user.value.values, index=df_user.key).to_dict()
    except Exception as e:
        print(f"Could not load rails file. Using empty dict. Error: {e}")
        return {}
        
# --- DECORADORES CORREGIDOS CON TEXTO FIJO ---
@st.cache_resource(show_spinner="Loading embeddings...")
def load_embedding():
    print("load_embedding")
    return OpenAIEmbeddings(openai_api_key=st.secrets.get("OPENAI_API_KEY"))

@st.cache_resource(show_spinner="Loading vector store...")
def load_vectorstore(username, _embedding): # Added underscore to ignore embedding object
    print(f"load_vectorstore for {username}")
    return AstraDB(
        embedding=_embedding,
        collection_name=f"vector_context_{username}",
        token=st.secrets["ASTRA_TOKEN"],
        api_endpoint=st.secrets.get("ASTRA_ENDPOINT", os.environ.get("ASTRA_ENDPOINT")) # Allow ENV fallback
    )

@st.cache_resource(show_spinner="Loading chat history...")
def load_chat_history(username, session_id): # Added session_id parameter
    print(f"load_chat_history for {username}_{session_id}")
    return AstraDBChatMessageHistory(
        session_id=f"{username}_{str(session_id)}",
        api_endpoint=st.secrets.get("ASTRA_ENDPOINT", os.environ.get("ASTRA_ENDPOINT")),
        token=st.secrets["ASTRA_TOKEN"],
    )

#################
### MAIN APP ####
#################

# --- LOGIN ---
# Minimal lang_dict for login screen
if 'lang_dict' not in st.session_state:
    st.session_state.lang_dict = load_localization(st.secrets.get("languages", {}).get("default", "es_ES"))

if not check_password():
    st.stop()  # Do not continue if check_password is not True.

# --- INITIALIZATION (POST-LOGIN) ---
username = st.session_state.user
language = st.secrets.get("languages", {}).get(username, "es_ES") # Default to Spanish
lang_dict = load_localization(language) # Load the full, correct language dict
st.session_state.lang_dict = lang_dict # Update session state

# Initialize resources
embedding = load_embedding()
if embedding:
    vectorstore = load_vectorstore(username, embedding)
if vectorstore:
    chat_history = load_chat_history(username, st.session_state.session_id)
else:
    chat_history = None

# --- SESSION STATE (POST-LOGIN & RESOURCES) ---
if 'messages' not in st.session_state:
    st.session_state.messages = [AIMessage(content=lang_dict.get('assistant_welcome', 'Welcome!'))]

# --- MAIN PAGE & SIDEBAR ---
try:
    welcome_file = Path(f"./customizations/welcome/{username}.md")
    if welcome_file.is_file():
        st.markdown(welcome_file.read_text(encoding='utf-8'))
    else:
        st.markdown(Path('./customizations/welcome/default.md').read_text(encoding='utf-8'))
except Exception:
    st.markdown(lang_dict.get('assistant_welcome', 'Welcome!'))

with st.sidebar:
    # Logo
    try:
        user_logo = Path(f"./customizations/logo/{username}.svg")
        if not user_logo.is_file(): user_logo = Path(f"./customizations/logo/{username}.png")
        if not user_logo.is_file(): user_logo = Path('./customizations/logo/default.svg')
        st.image(str(user_logo), use_column_width="always")
    except Exception as e:
        print(f"Error loading logo: {e}")
    st.text('')
    
    # Logout
    st.markdown(f"""{lang_dict.get('logout_caption','Logged in as')} :orange[{username}]""")
    if st.button(lang_dict.get('logout_button','Logout')):
        logout()
    st.divider()

    # Options Panel
    rails_dict = load_rails(username)
    disable_chat_history = st.toggle(lang_dict.get('disable_chat_history', "Disable Chat History"))
    top_k_history = st.slider(lang_dict.get('k_chat_history', "K for Chat History"), 1, 50, 5, disabled=disable_chat_history)
    memory = load_memory(chat_history, top_k_history if not disable_chat_history else 0) if chat_history else None

    if memory:
        if st.button(lang_dict.get('delete_chat_history_button', "Delete Chat History"), disabled=disable_chat_history):
            with st.spinner(lang_dict.get('deleting_chat_history', "Deleting...")):
                memory.clear()
            st.rerun()

    disable_vector_store = st.toggle(lang_dict.get('disable_vector_store', "Disable Vector Store"))
    top_k_vectorstore = st.slider(lang_dict.get('top_k_vector_store', "Top-K for Vector Store"), 1, 50, 5, disabled=disable_vector_store)
    strategy = st.selectbox(lang_dict.get('rag_strategy', "RAG Strategy"), ('Basic Retrieval', 'Maximal Marginal Relevance', 'Fusion'), help=lang_dict.get('rag_strategy_help', "Help"), disabled=disable_vector_store)

    custom_prompt_text = ''
    custom_prompt_index = 0
    try:
        custom_prompt_file = Path(f"""./customizations/prompt/{username}.txt""")
        if custom_prompt_file.is_file():
            custom_prompt_text = custom_prompt_file.read_text(encoding='utf-8')
            custom_prompt_index = 2 # 'Custom'
        else:
            custom_prompt_text = Path(f"""./customizations/prompt/default.txt""").read_text(encoding='utf-8')
            custom_prompt_index = 0
    except Exception as e:
        print(f"Error loading prompt text: {e}")
        custom_prompt_text = "Prompt file not found."

    prompt_type = st.selectbox(lang_dict.get('system_prompt', "System Prompt"), ('Short results', 'Extended results', 'Custom'), index=custom_prompt_index)
    custom_prompt = st.text_area(lang_dict.get('custom_prompt', "Custom Prompt"), custom_prompt_text, help=lang_dict.get('custom_prompt_help', "Help"), disabled=(prompt_type != 'Custom'))
    print(f"""{disable_vector_store}, {top_k_history}, {top_k_vectorstore}, {strategy}, {prompt_type}""")
    st.divider()

    # --- BLOQUES DE INGESTA COMENTADOS ---
    # # Include the upload form for new data to be Vectorized
    # st.subheader(lang_dict.get('upload_header', "Upload Documents"))
    # uploaded_files = st.file_uploader(lang_dict.get('load_context', "Upload Files"), type=['txt', 'pdf', 'csv'], accept_multiple_files=True)
    # if st.button(lang_dict.get('load_context_button', "Process Files")):
    #     if uploaded_files and vectorstore:
    #         vectorize_text(uploaded_files, vectorstore, lang_dict)

    # # Include the upload form for URLs be Vectorized
    # st.subheader(lang_dict.get('url_header', "Load from URLs"))
    # urls_text = st.text_area(lang_dict.get('load_from_urls', "Enter URLs"), help=lang_dict.get('load_from_urls_help', "Help"))
    # if st.button(lang_dict.get('load_from_urls_button', "Process URLs")):
    #     urls = [url.strip() for url in urls_text.split(',') if url.strip()]
    #     if urls and vectorstore:
    #         vectorize_url(urls, vectorstore, lang_dict)
    
    # # Drop the vector data and start from scratch
    # delete_is_enabled = str(st.secrets.get("delete_option", {}).get(username, "False")).lower() == 'true'
    # if delete_is_enabled:
    #     st.divider()
    #     st.caption(lang_dict.get('delete_context', "Delete all context"))
    #     if st.button(lang_dict.get('delete_context_button', "⚠️ Delete Context")):
    #         if vectorstore and memory:
    #             with st.spinner(lang_dict.get('deleting_context', "Deleting...")):
    #                 vectorstore.clear()
    #                 memory.clear()
    #                 st.session_state.messages = [AIMessage(content=lang_dict.get('assistant_welcome', "Welcome!"))]
    #             st.rerun()
    #         else:
    #             st.warning("Cannot delete, resources not loaded.")
    # --- FIN DE BLOQUES COMENTADOS ---

    st.divider()

    # --- BLOQUE DE CÁMARA COMENTADO ---
    # # st.subheader(lang_dict.get('camera_header', "Query with Image"))
    # # picture = st.camera_input(lang_dict.get('take_picture', "Take a picture"))
    # # if picture:
    # #     if st.secrets.get("OPENAI_API_KEY"):
    # #         response = describeImage(picture.getvalue(), language)
    # #         if response and response.choices and response.choices[0].message.content:
    # #             picture_desc = response.choices[0].message.content
    # #             # This logic needs to be integrated with the main chat input below
    # #             st.session_state.question_from_camera = picture_desc
    # #             st.rerun()
    # #     else:
    # #         st.warning("OpenAI Key not set, camera feature disabled.")
    # --- FIN DE BLOQUE DE CÁMARA COMENTADO ---
    st.divider()
    st.caption("v231227.01_mod")


# Draw rails
with st.sidebar:
        st.subheader(lang_dict.get('rails_1', "Suggestions"))
        st.caption(lang_dict.get('rails_2', "Try asking:"))
        if rails_dict:
            for i in sorted(rails_dict.keys()): # Sort for consistency
                st.markdown(f"{i}. {rails_dict[i]}")

# --- CHAT INTERFACE ---
# Draw all messages
for message in st.session_state.messages:
    st.chat_message(message.type).markdown(message.content)

# Get a prompt from a user
question = st.chat_input(lang_dict.get('assistant_question', "Your question..."))

if question:
    print(f"Got question: {question}")
    st.session_state.messages.append(HumanMessage(content=question))
    with st.chat_message('human'):
        st.markdown(question)

    with st.chat_message('assistant'):
        response_placeholder = st.empty()
        
        # This check ensures we don't proceed if resources failed to load
        if not model or (not disable_vector_store and not vectorstore):
            response_placeholder.markdown("Sorry, the assistant is not fully configured. Please check API keys and settings.")
        else:
            relevant_documents = []
            if not disable_vector_store:
                retriever = load_retriever(vectorstore, top_k_vectorstore)
                if strategy == 'Basic Retrieval':
                    relevant_documents = retriever.get_relevant_documents(query=question)
                elif strategy == 'Maximal Marginal Relevance':
                    relevant_documents = vectorstore.max_marginal_relevance_search(query=question, k=top_k_vectorstore)
                elif strategy == 'Fusion':
                    generate_queries = generate_queries_chain(model, language)
                    fusion_queries = generate_queries.invoke({"original_query": question})
                    
                    content_fusion = f"""*{lang_dict.get('using_fusion_queries','Using fusion queries...')}*"""
                    for fq in fusion_queries: content_fusion += f"""\n📙 :orange[{fq}]"""
                    st.session_state.messages.append(AIMessage(content=content_fusion))
                    st.markdown(content_fusion) # Show intermediate step

                    chain_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion
                    fused_results = chain_fusion.invoke({"original_query": question})
                    relevant_documents = [doc_tuple[0] for doc_tuple in fused_results][:top_k_vectorstore]

            history = memory.load_memory_variables({}) if memory else {"chat_history": []}
            
            inputs_map = RunnableMap({
                'context': lambda x: x['context'],
                'chat_history': lambda x: x['chat_history'],
                'question': lambda x: x['question']
            })
            chain = inputs_map | get_prompt(prompt_type, custom_prompt, language) | model
            response = chain.invoke(
                {'question': question, 'chat_history': history.get('chat_history', []), 'context': relevant_documents},
                config={'callbacks': [StreamHandler(response_placeholder)]}
            )
            
            final_content = response.content
            if memory:
                memory.save_context({'question': question}, {'answer': final_content})

            if not disable_vector_store:
                final_content += f"\n\n*{lang_dict.get('sources_used','Sources:')}*"
                sources_used = []
                for doc in relevant_documents:
                    source_name = doc.metadata.get('source', 'Unknown')
                    source_basename = os.path.basename(os.path.normpath(source_name))
                    if source_basename not in sources_used:
                        final_content += f"\n📙 :orange[{source_basename}]"
                        sources_used.append(source_basename)

            if not disable_chat_history:
                history_len = len(history.get('chat_history', [])) // 2
                final_content += f"\n\n*{lang_dict.get('chat_history_used','History used')}: ({history_len}/{top_k_history})*"

            response_placeholder.markdown(final_content)
            st.session_state.messages.append(AIMessage(content=final_content))
