# streamlit_app.py
import os
import streamlit as st

# LangChain + vectorstore
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Hugging Face / LangChain LLM wrappers
from langchain import HuggingFaceHub
from huggingface_hub import InferenceClient

# ----------------------------
# Config
# ----------------------------
DB_FAISS_PATH = "vectorstore/db_faiss"
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
HF_TOKEN = st.secrets.get("HF_TOKEN")  # or os.environ.get("HF_TOKEN")

# quick debug
st.write("HF_TOKEN present?", bool(HF_TOKEN))
st.write("HUGGINGFACE_REPO_ID:", HUGGINGFACE_REPO_ID)

# ----------------------------
# Helper: Vectorstore
# ----------------------------
def get_vectorstore():
    """Load the FAISS vector store with the sentence-transformer embedding model."""
    # using langchain_huggingface embeddings (no model weights loaded, just remote embedder)
    from langchain_huggingface import HuggingFaceEmbeddings
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

# ----------------------------
# Helper: LLM loader (HuggingFaceHub primary, InferenceClient fallback)
# ----------------------------
def load_llm_hfhub(repo_id: str, token: str, task: str = "text-generation"):
    """
    Create LangChain HuggingFaceHub LLM wrapper with explicit task.
    This uses Hugging Face inference (remote) and should not require local tokenizer installs.
    """
    return HuggingFaceHub(
        repo_id=repo_id,
        huggingfacehub_api_token=token,
        task=task,
        model_kwargs={"temperature": 0.2, "max_new_tokens": 512},
    )

def load_llm_inferenceclient(repo_id: str, token: str):
    """
    Simple fallback wrapper around huggingface_hub.InferenceClient.
    Provides a minimal callable interface .generate(prompt) or __call__(prompt).
    """
    client = InferenceClient(token=token)

    class SimpleHFClient:
        def __init__(self, client, model):
            self.client = client
            self.model = model

        def __call__(self, prompt: str):
            # Use the client method available in modern huggingface_hub
            resp = self.client.text_generation(model=self.model, inputs=prompt, max_new_tokens=256, temperature=0.2)
            # Normalize typical response shapes to a simple string
            try:
                if isinstance(resp, list) and len(resp) and isinstance(resp[0], dict):
                    return resp[0].get("generated_text") or str(resp[0])
                return str(resp)
            except Exception:
                return str(resp)

        # Provide compatibility for some LangChain wrappers expecting .generate or .predict
        def generate(self, inputs):
            # inputs: list of dicts or list[str] depending on caller; keep minimal
            if isinstance(inputs, list):
                text = self.__call__(inputs[0])
            elif isinstance(inputs, dict):
                # { "prompt": "..." } or {"input": "..." }
                text = self.__call__(inputs.get("prompt") or inputs.get("input") or "")
            else:
                text = self.__call__(str(inputs))
            return text

    return SimpleHFClient(client, repo_id)

def load_llm(repo_id: str, token: str):
    """
    Primary entry point used by the app. Try HFHub first, then fallback to direct InferenceClient wrapper.
    """
    # prefer HuggingFaceHub (clean remote inference path)
    try:
        llm = load_llm_hfhub(repo_id, token, task="text-generation")
        st.write("Using HuggingFaceHub wrapper for LLM.")
        return llm
    except Exception as e:
        st.warning("HuggingFaceHub wrapper failed, falling back to InferenceClient. See logs.")
        st.write("HuggingFaceHub error:", e)
        return load_llm_inferenceclient(repo_id, token)

# ----------------------------
# Prompt builder
# ----------------------------
def build_prompt() -> PromptTemplate:
    template = (
        """
        You are an assistant for medical question-answering tasks. Use the retrieved context pieces to answer the question.
        If the answer is not contained in the context, simply say you do not know.
        • Answer in concise bullet-points.
        • Cite the page number after each bullet like (p-X).
        • Only use the provided context.
        
        Question: {question}
        Context:
        {context}
        """
    )
    return PromptTemplate(template=template, input_variables=["context", "question"])

# ----------------------------
# Session state
# ----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ----------------------------
# Sidebar – user input
# ----------------------------
with st.sidebar:
    st.header("💬 Chat Interface")
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_area("Ask a question:", placeholder="Type your question here…")
        submitted = st.form_submit_button("Submit")

    if submitted and user_input:
        with st.spinner("Generating answer…"):
            try:
                # append user message
                st.session_state.chat_history.append({"role": "user", "content": user_input})

                # 1) vectorstore
                vectorstore = get_vectorstore()

                # 2) llm (hf hub or fallback)
                llm_obj = load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN)

                # 3) build retrieval chain (use RetrievalQA which is stable)
                retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm_obj,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": build_prompt()},
                )

                # 4) run the chain (handle different chain interfaces)
                resp = None
                try:
                    # prefer calling as a Mapping (many langchain versions)
                    resp = qa_chain({"query": user_input})
                except TypeError:
                    try:
                        # some versions use .invoke
                        resp = qa_chain.invoke({"query": user_input})
                    except Exception:
                        try:
                            # others expect .run which returns a string (no source docs)
                            text = qa_chain.run(user_input)
                            resp = {"result": text, "source_documents": []}
                        except Exception as e:
                            raise RuntimeError("Failed to execute QA chain: " + str(e)) from e

                # 5) interpret response robustly
                # possible shapes:
                # - dict with keys like "result" and "source_documents"
                # - dict with "output_text" or "answer"
                # - plain string (already converted above to dict)
                answer = ""
                docs = []
                if isinstance(resp, dict):
                    # try common keys
                    if "result" in resp:
                        answer = resp.get("result") or ""
                    elif "answer" in resp:
                        answer = resp.get("answer") or ""
                    elif "output_text" in resp:
                        answer = resp.get("output_text") or ""
                    else:
                        # fallback: join values
                        answer = str(resp)
                    docs = resp.get("source_documents") or resp.get("source_documents", []) or []
                else:
                    answer = str(resp)
                    docs = []

                # 6) format source chunks
                source_lines = []
                for d in docs:
                    try:
                        page = d.metadata.get("page", "?")
                        source_lines.append(f"- p-{page}: {d.page_content.strip()}")
                    except Exception:
                        # if doc shape unexpected
                        source_lines.append(f"- {str(d)[:200]}")

                formatted_answer = answer + "\n\n**Source Chunks:**\n" + ("\n".join(source_lines) if source_lines else "None")

                # 7) append assistant response
                st.session_state.chat_history.append({"role": "assistant", "content": formatted_answer})

            except Exception as e:
                st.error(f"❌ {e}")

# ----------------------------
# Main – conversation history
# ----------------------------
st.title("❄️ Medical Chatbot")
st.markdown("## 📜 Conversation History")
if st.session_state.chat_history:
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"**🧑‍💻 User:** {msg['content']}")
        else:
            st.markdown(f"**🤖 Assistant:** {msg['content']}")
        st.markdown("---")
else:
    st.info("No conversations yet. Start by asking a question in the sidebar!")

