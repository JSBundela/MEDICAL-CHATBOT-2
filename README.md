

```markdown
# 🩺 Medical Chatbot using LangChain, FAISS & HuggingFace

This project is a **Streamlit-based medical question-answering chatbot** powered by:
- **FAISS** for fast semantic retrieval
- **HuggingFace** large language model (`mistralai/Mistral-7B-Instruct-v0.3`)
- **LangChain** for chaining the components together

It allows users to ask medical questions and get **contextually accurate** and **source-cited** answers derived from a custom document corpus.

---

## 🚀 Features

- ⚡ Fast document retrieval with FAISS
- 🧠 Natural language understanding via Mistral-7B LLM (HuggingFace)
- 🔎 Source-based citation with page numbers
- 🗃️ Persistent chat history via Streamlit session state
- 🛠️ Easy to extend for other domains

---

## 📁 Project Structure

```
.
├── app.py                      # Main Streamlit app
├── vectorstore/
│   └── db_faiss/              # FAISS index directory
├── requirements.txt           # Python dependencies
└── README.md                  # You're here!
```

---

## 📦 Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/medical-chatbot.git
cd medical-chatbot
```

2. **Create virtual environment and install dependencies**
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

3. **Set up your HuggingFace token**
```bash
export HF_TOKEN=your_huggingface_token_here
```
> Or set it inside your `.env` file and load it manually in `app.py`

4. **Ensure FAISS vectorstore is populated**
> Make sure the `vectorstore/db_faiss/` directory contains your pre-built FAISS index with document chunks and metadata.

---

## ▶️ Running the App

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## 🛠️ How it Works

1. **User inputs a medical question**
2. **FAISS** retrieves top-k relevant chunks from the document vector store
3. **LangChain** chains the prompt and retrieved context to query the Mistral-7B model
4. **The LLM generates an answer**, citing the source page numbers
5. **Streamlit displays the full chat history**

---

## 🧠 Model Details

- **Retriever**: `sentence-transformers/all-MiniLM-L6-v2`
- **LLM**: [`mistralai/Mistral-7B-Instruct-v0.3`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
- **Prompting**: Customized with bullet-point style and source citation requirement

---

## 🧪 Example Questions

- *"What are the common symptoms of diabetes?"*
- *"How does hypertension affect the kidneys?"*
- *"List treatment options for prostate cancer."*

---

## 📌 TODOs / Improvements

- [ ] Add support for document upload and dynamic FAISS indexing
- [ ] Improve UI with custom Streamlit components
- [ ] Add session history export (PDF/CSV)
- [ ] Integrate user feedback loop

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or pull requests.

---

## 📜 License

MIT License © 2025 [Your Name]

---




Let me know if you want this customized with your actual GitHub username, project title, or any more sections like video demo, deployment instructions, etc.
