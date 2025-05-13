import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
#from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

HF_TOKEN = st.secrets["HF_TOKEN"]



# ----------------------------
# Configuration
# ----------------------------
DB_FAISS_PATH = "vectorstore/db_faiss"
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
HF_TOKEN = os.environ.get("HF_TOKEN")

# ----------------------------
# Helper Functions
# ----------------------------

def get_vectorstore():
    """Load the FAISS vector store with the sentence‑transformer embedding model."""
    from langchain_huggingface import HuggingFaceEndpoint 
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def load_llm(repo_id: str, token: str):
    """Return a HuggingFace endpoint with sensible defaults."""
    from langchain_huggingface import HuggingFaceEndpoint 
    return HuggingFaceEndpoint(
        model=repo_id,
        temperature=0.5,
        task="conversational",
        huggingfacehub_api_token=token,
        max_new_tokens=512
        #model_kwargs={"max_length": 512}
        #model_kwargs={"max_new_tokens": 512} 
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
                #vectorstore = get_vectorstore()

                from langchain.chains import ConversationalRetrievalChain

                # 2️⃣ Build QA chain
                vectorstore = get_vectorstore()
                qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN),
                    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
                    return_source_documents=True,
                    combine_docs_chain_kwargs={"prompt": build_prompt()},
                )

               

                # 3️⃣  Run chain
                #resp = qa_chain.invoke({"query": user_input})
                # Run with both keys
                resp = qa_chain({
                    "question": user_input,
                    "chat_history": st.session_state.chat_history[:-1]  # exclude the blank assistant turn
                })
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
