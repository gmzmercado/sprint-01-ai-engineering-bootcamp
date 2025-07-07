import streamlit as st
from core.config import config
from openai import OpenAI
from groq import Groq
from google import genai
from qdrant_client import QdrantClient

# NEW: Import the RAG pipeline
from retrieval import rag_pipeline

qdrant_client = QdrantClient(
    url=f"http://{config.QDRANT_URL}:6333",
)



## Let's create a sidebar with a dropdown for the model list and providers
with st.sidebar:
    st.title("Settings")

    # Dropdown for the provider and model list
    provider = st.selectbox("Provider", ["OpenAI", "Groq", "Google"], key="provider_select")
    if provider == "OpenAI":
        model_name = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"], key="model_select")
    elif provider == "Groq":
        model_name = st.selectbox("Model", ["llama-3.3-70b-versatile"], key="model_select")
    elif provider == "Google":
        model_name = st.selectbox("Model", ["gemini-2.0-flash"], key="model_select")

    # Temperature slider with collapsible explanation
    with st.expander("ℹ️ About Temperature", expanded=False):
        st.markdown(
            "**Temperature** controls the randomness of the model's output. "
            "Higher values make output more creative, lower values make it more deterministic."
        )
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.05, key="temperature_slider")

    # Max tokens slider with collapsible explanation
    with st.expander("ℹ️ About Max Tokens", expanded=False):
        st.markdown(
            "**Max Tokens** sets the maximum number of tokens the model can generate. "
            "Higher values allow longer responses."
        )
    max_tokens = st.slider("Max Tokens", min_value=100, max_value=2000, value=500, step=50, key="max_tokens_slider")

    # Add Reset Chat button
    if st.button("🔄 Reset Chat"):
        # Use Streamlit's HTML component to execute JavaScript
        st.components.v1.html(
            """
            <script>
            window.parent.location.reload();
            </script>
            """,
            height=0,
        )

    # Save provider, model, temperature, and max_tokens to session state
    st.session_state.provider = provider
    st.session_state.model_name = model_name
    st.session_state.temperature = temperature
    st.session_state.max_tokens = max_tokens

# Ask for API key respective to the user's selection
if st.session_state.provider == "OpenAI":
    client = OpenAI(api_key=config.OPENAI_API_KEY)
elif st.session_state.provider == "Groq":
    client = Groq(api_key=config.GROQ_API_KEY)
elif st.session_state.provider == "Google":
    client = genai.Client(api_key=config.GOOGLE_API_KEY)

# Define a function called run_llm with error handling
def run_llm(client, messages, temperature=0.5, max_tokens=500):
    try:
        if st.session_state.provider == "Google":
            return client.models.generate_content(
                model=st.session_state.model_name,
                contents=[message["content"] for message in messages]
            ).text
        else:
            return client.chat.completions.create(
                model=st.session_state.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            ).choices[0].message.content
    except Exception as e:
        return f"❌ Error: {str(e)}"

# First steps here.
# client = openai.OpenAI(api_key=config.OPENAI_API_KEY)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You should never disclose what model you are based on."},
        {"role": "assistant", "content": "Hello! How can I assist you today?"}
        ]

for message in st.session_state.messages:
    if message["role"] != "system":  # Hide system messages from display
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("Hello, how can I assist you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        # output = run_llm(
        #     client,
        #     st.session_state.messages,
        #     temperature=st.session_state.temperature,
        #     max_tokens=st.session_state.max_tokens
        # )
        output = rag_pipeline(prompt, qdrant_client)

        if output:
            st.write(output["answer"])
            st.session_state.messages.append({"role": "assistant", "content": output})
        else:
            st.write("No response.")