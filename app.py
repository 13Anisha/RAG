import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
import os


groq_api_key = st.secrets["groq_api_key"]


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def get_chat_history():
    return st.session_state.get("chat_history", [])

def format_history(messages):
    return "\n".join(
        [f"User: {m.content}" if isinstance(m, HumanMessage) else f"Assistant: {m.content}" for m in messages]
    )

def fetch_transcript(video_id: str):
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en", "en-US"])
        return " ".join(chunk["text"] for chunk in transcript_list)
    except TranscriptsDisabled:
        st.error("Transcript is disabled for this video.")
        return None
    except Exception as e:
        st.error(f"Error fetching transcript: {str(e)}")
        return None

def process_video(video_id: str):
    index_path = f"faiss_index/{video_id}"
    faiss_file = os.path.join(index_path, "index.faiss")
    pkl_file = os.path.join(index_path, "index.pkl")

    
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

   
    if os.path.exists(faiss_file) and os.path.exists(pkl_file):
        vectorstore = FAISS.load_local(
            index_path,
            embeddings=embed_model,
            index_name="index",
            allow_dangerous_deserialization=True 
        )
        print(f"Loaded FAISS index for video ID: {video_id}")
    else:
        print(f"Generating FAISS index for video ID: {video_id}")
        transcript = fetch_transcript(video_id)
        if transcript is None:
            return None

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.create_documents([transcript], metadatas=[{"video_id": video_id}])

        vectorstore = FAISS.from_documents(documents=chunks, embedding=embed_model)
        vectorstore.save_local(index_path, index_name="index")
        print(f"Saved FAISS index at {index_path}")

    return vectorstore.as_retriever(search_kwargs={"k": 5})



st.set_page_config(page_title="RAG Video Assistant", layout="centered")
st.title("🎥 RAG Based Video Assistant")

video_id = st.text_input("Enter YouTube Video ID (e.g., 0AbKu6rqGZo):")

if video_id:
    with st.spinner("Processing... Please wait."):

        retriever = process_video(video_id)

    if retriever:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a knowledgeable video assistant. Use only the information provided in the context and chat history to answer the QUESTION.

Instructions:
- If the answer is clearly present in the context or chat history, provide a direct and complete response.
- If the context and history do **not** contain enough information, reply with:
  "The context does not contain enough information. However, here's a general answer:"
  ...and then provide a helpful general response to the user.
- Do **not** make up answers not supported by the context or chat history unless explicitly allowed.
- If the question is vague (e.g., "Can you elaborate?", "What do you mean by that?", "Please explain more"), try to infer what the user is referring to based on the last interaction in the chat history.
- If the vague question cannot be reasonably inferred, ask the user to clarify their question politely."""),
            ("user", "{question}"),
            ("system", "Context:\n{context}"),
            ("system", "Previous Chat:\n{chat_history}")
        ])

        llm = ChatGroq(
            temperature=0.4,
            groq_api_key=groq_api_key,
            model_name="llama-3.1-8b-instant"
        )

        chain = (
            {
                "context": retriever | RunnableLambda(lambda docs: "\n\n".join([doc.page_content for doc in docs])),
                "question": RunnablePassthrough(),
                "chat_history": RunnableLambda(lambda _: format_history(get_chat_history()))
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        
        for message in st.session_state.chat_history:
            role = "user" if isinstance(message, HumanMessage) else "assistant"
            st.chat_message(role).write(message.content)

        user_input = st.chat_input("Ask something about the video...")
        if user_input:
            st.session_state.chat_history.append(HumanMessage(content=user_input))
            response = chain.invoke(user_input)
            st.session_state.chat_history.append(AIMessage(content=response))
            st.chat_message("user").write(user_input)
            st.chat_message("assistant").write(response)

print(get_chat_history())
