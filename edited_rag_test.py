import streamlit as st
from loguru import logger

from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory, StreamlitChatMessageHistory

from langchain_google_genai import ChatGoogleGenerativeAI
import os

def main():
    st.set_page_config(page_title="Streamlit_Rag", page_icon=":books:")
    st.title("_Private Data :red[Q/A Chat]_ :books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        uploaded_files = st.file_uploader("Upload your file", type=['pdf','docx','pptx'], accept_multiple_files=True)
        google_api_key = st.text_input("Google API Key", key="chatbot_api_key", type="password")
        os.environ["GOOGLE_API_KEY"] = google_api_key
        process = st.button("Process")

    if process:
        if not google_api_key:
            st.info("Please add your Google API key to continue.")
            st.stop()

        docs = get_text(uploaded_files)
        chunks = get_text_chunks(docs)
        vectordb = get_vectorstore(chunks)
        st.session_state.conversation = get_conversation_chain(vectordb)
        st.session_state.processComplete = True

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{
            "role": "assistant",
            "content": "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"
        }]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    StreamlitChatMessageHistory(key="chat_messages")

    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        with st.chat_message("assistant"):
            chain = st.session_state.conversation
            with st.spinner("Thinking..."):
                result = chain({"question": query})
                st.session_state.chat_history = result['chat_history']
                response = result['answer']
                sources = result['source_documents']

                st.markdown(response)
                with st.expander("참고 문서 확인"):
                    for doc in sources[:3]:
                        st.markdown(f"- {doc.metadata['source']}")

        st.session_state.messages.append({"role": "assistant", "content": response})


def get_text(docs):
    all_docs = []
    for doc in docs:
        fname = doc.name
        with open(fname, "wb") as f:
            f.write(doc.getvalue())
            logger.info(f"Uploaded {fname}")

        if fname.lower().endswith(".pdf"):
            loader = PyPDFLoader(fname)
        elif fname.lower().endswith(".docx"):
            loader = Docx2txtLoader(fname)
        elif fname.lower().endswith(".pptx"):
            loader = UnstructuredPowerPointLoader(fname)
        else:
            continue

        all_docs.extend(loader.load_and_split())
    return all_docs


def get_text_chunks(docs):
    # 토큰 대신 문자 수 기반 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200
    )
    return text_splitter.split_documents(docs)


def get_vectorstore(chunks):
    # SentenceTransformer 임베딩 사용
    embeddings = SentenceTransformerEmbeddings(
        model_name="jhgan/ko-sroberta-multitask"
    )
    # pure-Python Chroma 사용
    vectordb = Chroma.from_documents(chunks, embeddings)
    return vectordb


def get_conversation_chain(vectordb):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_type='mmr'),
        memory=ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True,
            output_key='answer'
        ),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True
    )


if __name__ == '__main__':
    main()
