import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace
from langchain.chains import RetrievalQA
#from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

HF_TOKEN = st.secrets["HF_TOKEN"]


import streamlit as st, sys, traceback
st.write("sys.executable:", sys.executable)
try:
    import torch
    st.write("torch import OK")
    st.write("torch.__version__:", getattr(torch, "__version__", None))
    st.write("torch.__file__:", getattr(torch, "__file__", None))
    try:
        st.write("torch.version.cuda:", torch.version.cuda)
    except Exception as ex:
        st.write("couldn't get torch.version.cuda:", ex)
    try:
        st.write("torch.cuda.is_available():", torch.cuda.is_available())
    except Exception as ex:
        st.write("couldn't query torch.cuda.is_available():", ex)
except Exception:
    st.error("torch import FAILED — traceback below")
    st.code(traceback.format_exc())

try:
    import sentence_transformers as s
    st.write("sentence_transformers import OK:", getattr(s, "__file__", None))
except Exception:
    st.error("sentence_transformers import FAILED — traceback below")
    st.code(traceback.format_exc())
from huggingface_hub.utils._errors import HfHubHTTPError
from huggingface_hub import InferenceClient

client = InferenceClient(token=HF_TOKEN)

# Primary model (the one that returned 410)
PRIMARY_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

# Fallback candidates — edit this list to models you want to allow
FALLBACK_MODELS = [
    "tiiuae/falcon-7b-instruct",
    "google/flan-t5-large",     # not chat-style but OK for QA
    "facebook/galactica-1.3b"   # example; ensure availability
]

def hf_chat_with_fallback(message, primary=PRIMARY_MODEL, fallbacks=FALLBACK_MODELS, max_tokens=512):
    """Try primary model; on 410 (deprecated) try fallbacks in order."""
    tried = []
    # Helper to call the HF chat completion api correctly
    def call_chat(model_id):
        tried.append(model_id)
        # Use chat.completions.create(...) which works for current clients
        return client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": message}],
            max_tokens=max_tokens,
        )

    # Try primary + fallbacks
    for model in [primary] + list(fallbacks):
        try:
            resp = call_chat(model)
            # extract text robustly depending on response shape
            # many responses have .choices[0].message.content
            if hasattr(resp, "choices") and len(resp.choices) > 0:
                content = getattr(resp.choices[0].message, "content", None)
                if content:
                    return {"model": model, "text": content, "raw": resp}
            # sometimes returns dict-like
            if isinstance(resp, dict):
                try:
                    return {"model": model, "text": resp["choices"][0]["message"]["content"], "raw": resp}
                except Exception:
                    return {"model": model, "text": str(resp), "raw": resp}
            # last fallback
            return {"model": model, "text": str(resp), "raw": resp}
        except HfHubHTTPError as hf_err:
            # Inspect HTTP status code/message (410 -> model deprecated at provider)
            status = getattr(hf_err, "response", None)
            msg = str(hf_err)
            # If it's 410 (Gone) — try next fallback
            if "410" in msg or "Gone" in msg or "deprecated" in msg.lower():
                st.warning(f"Model {model} is deprecated/removed on its provider (HTTP 410). Trying next candidate...")
                continue
            # For non-410 HF errors, re-raise so it surfaces
            raise
        except Exception as e:
            # Unexpected error (network, auth) — rethrow or surface depending on your policy
            st.error(f"Error calling model {model}: {e}")
            raise

    # If we exhausted all candidates
    raise RuntimeError(f"All candidate models failed. Tried: {tried}")

# Usage example inside your Streamlit flow:
try:
    result = hf_chat_with_fallback("Hello, how are you?")
    st.write("Used model:", result["model"])
    st.write(result["text"])
except Exception as e:
    st.error(f"Model call failed: {e}")
    # Optionally show full traceback for debug
    import traceback; st.code(traceback.format_exc())



# ----------------------------
# Configuration
# ----------------------------
DB_FAISS_PATH = "vectorstore/db_faiss"
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
#HUGGINGFACE_REPO_ID ="tiiuae/falcon-7b-instruct"
#HUGGINGFACE_REPO_ID ="meta-llama/Llama-2-7b-chat-hf"

#HF_TOKEN = os.environ.get("HF_TOKEN")

# ----------------------------
# Helper Functions
# ----------------------------

def get_vectorstore():
    """Load the FAISS vector store with the sentence‑transformer embedding model."""
    from langchain_huggingface import HuggingFaceEndpoint 
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def load_endpoint(repo_id: str, token: str):
    """Return a HuggingFace endpoint with sensible defaults."""
    from langchain_huggingface import HuggingFaceEndpoint 
    return HuggingFaceEndpoint(
        model=repo_id, 
        #repo_id=repo_id,
        temperature=0.5,
        #task="text-generation",
        task="conversational",
        huggingfacehub_api_token=token,
        max_new_tokens=512
        #model_kwargs={"max_length": 512}
        #model_kwargs={"max_new_tokens": 512} 
    )

from langchain_huggingface import ChatHuggingFace

def load_llm(repo_id: str, token: str):
    endpoint = load_endpoint(repo_id, token)
    return ChatHuggingFace(
        llm=endpoint  # ⚠️ pass the endpoint here
    )

def build_prompt() -> PromptTemplate:
    template = (
        """
        You are an assistant for medical question‑answering tasks. Use the retrieved context pieces to answer the question.
        If the answer is not contained in the context, simply say you do not know.
        • Answer in concise bullet‑points.
        • Cite the page number after each bullet like (p‑X).
        • Only use the provided context.
        
        Question: {question}
        Context:
        {context}
        """
    )
    return PromptTemplate(template=template, input_variables=["context", "question"])


# ----------------------------
# Session State
# ----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list[dict[str,str]] – top→bottom order


# ----------------------------
# Sidebar – User Input
# ----------------------------
with st.sidebar:
    st.header("💬 Chat Interface")
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_area("Ask a question:", placeholder="Type your question here…")
        submitted = st.form_submit_button("Submit")

    if submitted and user_input:
        with st.spinner("Generating answer…"):
            try:
                # 1️⃣  Append user query
                st.session_state.chat_history.append({"role": "user", "content": user_input})

                # 2️⃣  Build QA chain

                vectorstore = get_vectorstore()
                qa_chain = RetrievalQA.from_chain_type(
                    llm=load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN),
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": build_prompt()},
                )

                # 3️⃣  Run chain

                resp = qa_chain.invoke({"query": user_input})
                answer = resp.get("result", "")
                docs = resp.get("source_documents", [])

                # 4️⃣  Assemble formatted answer with chunks & page numbers
                source_lines = []
                for d in docs:
                    page = d.metadata.get("page", "?")
                    source_lines.append(f"- p‑{page}: {d.page_content.strip()}")
                formatted_answer = answer + "\n\n**Source Chunks:**\n" + "\n".join(source_lines)

                # 5️⃣  Append assistant response
                st.session_state.chat_history.append({"role": "assistant", "content": formatted_answer})

            except Exception as e:
                st.error(f"❌ {e}")

# ----------------------------
# Main – Conversation History (top → bottom)
# ----------------------------
st.title("❄️ Medical Chatbot")

st.markdown("## 📜 Conversation History")
if st.session_state.chat_history:
    for msg in st.session_state.chat_history:  # natural order
        if msg["role"] == "user":
            st.markdown(f"**🧑‍💻 User:** {msg['content']}")
        else:
            st.markdown(f"**🤖 Assistant:** {msg['content']}")
        st.markdown("---")
else:
    st.info("No conversations yet. Start by asking a question in the sidebar!")
