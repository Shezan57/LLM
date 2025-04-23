# libraries for document loader
from typing import Any
from langchain_community.document_loaders. import PyPDFLoader, TextLoader,UnstructuredWordDocumentLoader, UnstructuredEPubLoader
import logging
import pathlib
from langchain.schema import Document

# Libraries for vector storage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.schema import BaseRetriever

# Libraries for Embedding Filter
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from sentence_transformers import SentenceTransformer

# Libraries for mechanism to create retriever
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import BaseRetriever
from langchain.chains.base import Chain
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
import os
import tempfile

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
# Libraries for streamlit
import streamlit as st
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

# Libraries for dotenv
from dotenv import load_dotenv
load_dotenv()


# Document Loader
class EpubReader(UnstructuredEPubLoader):
    def __init__(self, file_path: str):
        super().__init__(file_path, mode="elements", strategy="fast")
class DocumentLoader(object):
    """Loads in a document with a supported extension"""
    supported_extensions = {".pdf": PyPDFLoader,
                            ".txt": TextLoader,
                            ".docx": UnstructuredWordDocumentLoader,
                            ".epub": EpubReader,
                            ".doc": UnstructuredWordDocumentLoader}

def load_document(temp_filepath: str) -> list[Document]:
    """Load a file and return as a list of documents."""
    ext = pathlib.Path(temp_filepath).suffix
    loader = DocumentLoader.supported_extensions.get(ext)
    if loader is None:
        raise ValueError(f"Unsupported file extension: {ext}. Cannot load this type of file.")
    loader = loader(temp_filepath)
    docs = loader.load()
    logging.info(docs)
    return docs

# Vector Storages

def configure_retriver(docs: list[Document]) -> BaseRetriever:
    """Configure the retriever with the loaded documents."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)
    return vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4})


# Retriever Chain
def configure_chain(retriever: BaseRetriever) -> Chain:
    """Configure the chain with the retriever."""
    memory = ConversationBufferMemory(memory_key="chat_memory", return_messages=True)

    #Setup LLM and QA chain; set temparature low to keep hallucinations in check
    llm = ChatGroq(model = "deepseek-r1-distill-llama-70b", temparature=0, streaming=True)

    #passing in max_tokens to limit amont automatically
    # truncates the tokens when prompting your llm!
    return ConversationalRetrievalChain(retriever=retriever, memory=memory, llm=llm, max_tokens=4000)

# Retriever logic to pass documents to the retriever setup

def configure_qa_chain(uploaded_files):
    """Read documents, configure retriever, and configure chain."""
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        docs.extend(load_document(temp_filepath))

    retriever = configure_retriver(docs=docs)
    return configure_chain(retriever=retriever)

# For interface design I will use streamlit

st.set_page_config(page_title="Lnagchain: Chat With Documents", page_icon=":robot:")
st.title("Langchain: Chat With Documents")

uploaded_files = st.file_uploader("Upload a document",
                                  type=list(DocumentLoader.supported_extensions.keys())
                                  ,accept_multiple_files=True)
if not uploaded_files:
    st.info("Please upload a document to start chatting.")
    st.stop()

qa_chain = configure_qa_chain(uploaded_files=uploaded_files)
assistant = st.chat_message("assistant")
user_query = st.chat_input(placeholder="Ask me anything...!")

if user_query:
    stream_handler = StreamlitCallbackHandler(assistant)
    response = qa_chain.invoke(user_query, callbacks=[stream_handler])
    st.markdown(response)




