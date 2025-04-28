# libraries for document loader
import os
# Add these imports at the top
import nltk
import sys
import subprocess
import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader, UnstructuredEPubLoader
import logging
import pathlib
from langchain.schema import Document
from ebooklib import epub

# Libraries for vector storage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch


# Libraries for Embedding Filter
from langchain_community.embeddings import FastEmbedEmbeddings

# Libraries for mechanism to create retriever
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import BaseRetriever
from langchain.chains.base import Chain
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.memory import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Libraries for streamlit
import streamlit as st
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

# Libraries for dotenv
from dotenv import load_dotenv
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
# load_dotenv()
api_key = os.environ["GROQ_API_KEY"]


# Check and install required dependencies
def install_dependencies():
    try:
        # Download NLTK punkt tokenizer
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')

        # Check if python-docx is installed
        try:
            import docx
        except ImportError:
            print("Installing python-docx...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "python-docx"])
            print("Installing unstructured[docx]...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "unstructured[docx]"])
    except Exception as e:
        print(f"Error installing dependencies: {e}")


#Call this function before using document loaders
install_dependencies()

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
    ext = pathlib.Path(temp_filepath).suffix.lower()

    # Special case for EPUB
    if ext == ".epub":
        try:
            # Try to install dependencies if not present
            try:
                import ebooklib
                from bs4 import BeautifulSoup
            except ImportError:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "ebooklib beautifulsoup4"])
                import ebooklib
                from bs4 import BeautifulSoup


            # Extract text from EPUB
            book = epub.read_epub(temp_filepath)
            contents = []

            # Process each document item in the EPUB
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    text = soup.get_text()
                    contents.append(Document(page_content=text, metadata={"source": temp_filepath}))

            return contents
        except Exception as e:
            logging.error(f"Error processing EPUB: {str(e)}")
            raise RuntimeError(f"Error loading EPUB {temp_filepath}: {str(e)}")

    # For all other file types
    loader_class = DocumentLoader.supported_extensions.get(ext)
    if loader_class is None:
        raise ValueError(f"Unsupported file extension: {ext}. Cannot load this type of file.")

    # Rest of your document loading code...
    try:
        if ext == ".txt":
            # Your existing text file handling
            try:
                loader = loader_class(temp_filepath, encoding='utf-8')
                docs = loader.load()
            except UnicodeDecodeError:
                for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                    try:
                        loader = loader_class(temp_filepath, encoding=encoding)
                        docs = loader.load()
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise RuntimeError(f"Failed to decode {temp_filepath} with multiple encodings")
        else:
            loader = loader_class(temp_filepath)
            docs = loader.load()

        return docs
    except Exception as e:
        raise RuntimeError(f"Error loading {temp_filepath}: {str(e)}")

# Vector Storages

def configure_retriver(docs: list[Document]) -> BaseRetriever:
    """Configure the retriever with the loaded documents."""
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)
        return vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4})
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise



# Retriever Chain
def configure_chain(retriever: BaseRetriever) -> Chain:
    """Configure the chain with the retriever."""
    # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

    #Setup LLM and QA chain; set temparature low to keep hallucinations in check
    llm = ChatGroq(api_key=api_key, model="llama-3.3-70b-versatile", temperature=0.5, streaming=True) #deepseek-r1-distill-llama-70b

    # create the chain without memory parameter
    qa_chain = ConversationalRetrievalChain.from_llm(
        retriever=retriever,
        # memory=memory,
        llm=llm,
        return_source_documents=True
    )

    # Wrap with message history
    chain_with_history = RunnableWithMessageHistory(
        qa_chain,
        lambda session_id: st.session_state.chat_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    # return ConversationalRetrievalChain.from_llm(retriever=retriever, memory=memory, llm=llm)
    return chain_with_history
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

# Page setup
st.set_option('client.showErrorDetails', True)
st.set_page_config(page_title="Langchain: Chat With Documents", page_icon=":robot:")
st.title("Langchain: Chat With Documents")

# File upload section
uploaded_files = st.file_uploader(
    "Upload a document",
    type=list(DocumentLoader.supported_extensions.keys()),
    accept_multiple_files=True
)

# Stop execution if no files uploaded
if not uploaded_files:
    st.info("Please upload a document to start chatting.")
    st.stop()

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()

# Initialize QA chain
qa_chain = configure_qa_chain(uploaded_files=uploaded_files)

# Display existing chat messages
for message in st.session_state.chat_history.messages:
    role = "user" if message.type == "human" else "assistant"
    with st.chat_message(role):
        st.write(message.content)

# Chat input
user_query = st.chat_input(placeholder="Ask me anything...!")

# Process user input
if user_query:
    # Display user message
    with st.chat_message("user"):
        st.write(user_query)

    # Add to message history
    st.session_state.chat_history.add_user_message(user_query)

    # Create assistant message container
    with st.chat_message("assistant"):
        # Create a placeholder for persistent display
        message_placeholder = st.empty()

        try:
            # Show a spinner while processing
            with st.spinner("Thinking..."):
                stream_handler = StreamlitCallbackHandler(st.container())

                response = qa_chain.invoke(
                    {"question": user_query},
                    config={"configurable": {"session_id": "default"}},
                    callbacks=[stream_handler]
                )

            # Ensure answer is displayed and persisted
            if "answer" in response:
                answer = response["answer"]
                # Display final answer in the placeholder
                message_placeholder.markdown(answer)
                # Add to chat history
                st.session_state.chat_history.add_ai_message(answer)
            else:
                st.error("Response format was unexpected.")
                st.write(response)
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.exception(e)