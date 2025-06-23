import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.schema.runnable import RunnableMap
from langchain_groq import ChatGroq

groq_api_key = st.secrets["GROQ_API_KEY"]

st.set_page_config(page_title="YouTube RAG Assistant", layout="centered")
st.title("YouTube RAG Assistant")
st.markdown("Ask questions about any YouTube video (with English transcript).")

video_id = st.text_input("Enter YouTube Video ID (e.g., pBRSZBtirAk):")

if video_id:
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        transcript = " ".join(chunk["text"] for chunk in transcript_list)
    except (TranscriptsDisabled, NoTranscriptFound):
        st.error("Transcript not available in English or is disabled for this video.")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    llm = ChatGroq(
        temperature=0.2,
        model_name="meta-llama/llama-4-scout-17b-16e-instruct",
        api_key=groq_api_key
    )

    memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history",input_key="question")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use only the context below to answer."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}\n\nContext:\n{context}")
    ])

    qa_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

    st.divider()
    st.subheader("Ask Questions About the Video")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        memory.clear()
        st.rerun()

    user_input = st.text_input("Ask a question:", key="user_input")
    if user_input:
        docs = retriever.get_relevant_documents(user_input)
        context = "\n\n".join(doc.page_content for doc in docs)

        response = qa_chain.run({
            "question": user_input,
            "context": context
        })

        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Assistant", response))

    for speaker, msg in st.session_state.chat_history:
        st.markdown(f"**{speaker}:** {msg}")
