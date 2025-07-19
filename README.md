# RAG-Based YouTube Video Assistant

This project is a **Streamlit application** that uses **Retrieval-Augmented Generation (RAG)** to answer questions about a YouTube video based on its transcript. It combines **LangChain**, **FAISS**, **HuggingFace embeddings**, and **Groq's LLaMA 3.1 8B Instant model** for efficient retrieval and generation.

---

## **Features**

- Extracts transcript from any public YouTube video having English Transcripts
- Chunks and embeds transcript using HuggingFace embeddings
- Stores and retrieves data using FAISS for efficient similarity search
- Answers user queries using Groq’s **LLaMA-3.1-8B-Instant** model
- Maintains multi-turn conversation with chat history
- Caches transcript index locally for faster reuse

---

## **Tech Stack**

| Component              | Description                                    |
|------------------------|------------------------------------------------|
| **Streamlit**          | UI for interactive chat app                    |
| **LangChain**          | Manages prompt chains and retrieval logic     |
| **HuggingFace**        | Sentence embedding model                       |
| **FAISS**              | Vector store for fast similarity search        |
| **Groq**               | LLaMA-3.1-8B-Instant LLM for answering queries |
| **YouTube Transcript API** | Fetches transcript data from YouTube       |

---

## **Steps to Run the App**


## 1. Clone the Repository

Clone the repository to your local machine and navigate into the project directory:

git clone https://github.com/13Anisha/RAG

cd "your folder"

## 2. Install Dependencies
Make sure you have Python 3.9+ installed. Then, install the required packages:

pip install -r requirements.txt

## 3. Add Groq API Key

Create a .streamlit folder in the root directory (if it doesn’t already exist), and inside it, create a file named secrets.toml:

groq_api_key = "your_groq_api_key"

## 4. Run the Streamlit App

streamlit run app.py






