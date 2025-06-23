# 🧠 RAG Video Assistant

## App Link:https://sjdgkfxfetywxeupxcypbc.streamlit.app/

A Streamlit app that lets you interact with any **YouTube video** (with English transcript) using **RAG (Retrieval-Augmented Generation)**. Just enter a video ID, ask questions, and get context-aware answers powered by **LLMs via Groq**.

> 🔗 Live Repo: [github.com/13Anisha/RAG](https://github.com/13Anisha/RAG)

---

## 🚀 Features

- 🔎 Retrieve English transcript from a YouTube video
- 🧩 Split transcript into searchable chunks
- 💡 Ask questions about the video content
- 🗣️ Maintains conversation history
- ⚡ Fast, context-rich answers using Groq's LLaMA model

---

## 🛠️ Built With

- [Streamlit](https://streamlit.io/) – UI Framework
- [LangChain](https://www.langchain.com/) – Chains, Memory, Prompting
- [FAISS](https://github.com/facebookresearch/faiss) – Vector store
- [Hugging Face](https://huggingface.co/) – Embeddings
- [Groq](https://groq.com/) – LLM backend (`llama-4-scout`)
- [YouTube Transcript API](https://pypi.org/project/youtube-transcript-api/)

---

## 📦 Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/13Anisha/RAG.git
   cd RAG
2 **Set Up Virtual Environment**

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

3.**Install Requirements**

pip install -r requirements.txt

4.**Configure Secrets**

Create a file at .streamlit/secrets.toml:

GROQ_API_KEY = "your_groq_api_key_here"

5.**Run the App**

streamlit run app.py



## How It Works

1.User inputs a YouTube video ID

2.Transcript is fetched (English only)

3.Transcript is chunked and embedded

4.Relevant chunks are retrieved based on question

5.Prompt + memory are sent to Groq LLM via LangChain

6.Response is displayed in Streamlit interface

