import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import tempfile
import shutil

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="PDF Query App",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()
if 'upload_dir' not in st.session_state:
    st.session_state.upload_dir = "uploaded_documents"
    os.makedirs(st.session_state.upload_dir, exist_ok=True)

# Get API key from environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("⚠️ GOOGLE_API_KEY not found in .env file!")
    st.stop()

# Sidebar for PDF upload and processing
with st.sidebar:
    st.header("📄 Document Upload")
    
    uploaded_file = st.file_uploader(
        "Upload PDF Document",
        type=['pdf'],
        help="Upload a PDF file to process and store in the vector database"
    )
    
    if uploaded_file:
        if st.button("🚀 Proceed to Vectorize", use_container_width=True):
            with st.spinner("Processing document..."):
                try:
                    # Save uploaded file to local system
                    file_path = os.path.join(st.session_state.upload_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    st.info(f"✅ File saved: {uploaded_file.name}")
                    
                    # Load the PDF
                    loader = PyPDFLoader(file_path)
                    documents = loader.load()
                    
                    # Split documents into chunks
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200,
                        length_function=len
                    )
                    chunks = text_splitter.split_documents(documents)
                    
                    st.info(f"📄 Document split into {len(chunks)} chunks")
                    
                    # Create embeddings
                    embeddings = GoogleGenerativeAIEmbeddings(
                        model="models/embedding-001",
                        google_api_key=GOOGLE_API_KEY
                    )
                    
                    # Create or update ChromaDB vectorstore
                    persist_directory = "./chroma_db"
                    
                    if st.session_state.vectorstore is None:
                        st.session_state.vectorstore = Chroma.from_documents(
                            documents=chunks,
                            embedding=embeddings,
                            persist_directory=persist_directory
                        )
                    else:
                        # Add documents to existing vectorstore
                        st.session_state.vectorstore.add_documents(chunks)
                    
                    # Add to processed files
                    st.session_state.processed_files.add(uploaded_file.name)
                    
                    st.success(f"✅ Vectorization complete for {uploaded_file.name}!")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"❌ Error processing document: {str(e)}")
    
    # Display processed files
    if st.session_state.processed_files:
        st.divider()
        st.subheader("📚 Processed Documents")
        for file in st.session_state.processed_files:
            st.text(f"✓ {file}")

# Main app page
st.title("🤖 Chat with Your Documents")
st.markdown("Ask questions about your uploaded documents and get AI-powered answers!")

# Document selection for chat
st.subheader("📂 Select Documents to Chat With")

# Get list of uploaded files
uploaded_files_list = list(st.session_state.processed_files)

if uploaded_files_list:
    selected_docs = st.multiselect(
        "Choose one or more documents:",
        options=uploaded_files_list,
        default=uploaded_files_list,
        help="Select which documents to use for answering your questions"
    )
    
    if selected_docs and st.session_state.vectorstore:
        st.success(f"✅ Ready to chat with {len(selected_docs)} document(s)")
        
        # Chat interface
        st.divider()
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if query := st.chat_input("Ask a question about your documents..."):
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": query})
            
            with st.chat_message("user"):
                st.markdown(query)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Initialize Gemini LLM
                        llm = ChatGoogleGenerativeAI(
                            model="gemini-pro",
                            google_api_key=GOOGLE_API_KEY,
                            temperature=0.3
                        )
                        
                        # Create custom prompt template
                        prompt_template = """You are a helpful assistant that answers questions based ONLY on the provided context from the documents.
                        
If the answer cannot be found in the context, say "I cannot find information about this in the provided documents."

Do NOT use your own knowledge to answer questions. Only use the context provided below.

Context:
{context}

Question: {question}

Answer:"""
                        
                        PROMPT = PromptTemplate(
                            template=prompt_template,
                            input_variables=["context", "question"]
                        )
                        
                        # Create retrieval QA chain
                        qa_chain = RetrievalQA.from_chain_type(
                            llm=llm,
                            chain_type="stuff",
                            retriever=st.session_state.vectorstore.as_retriever(
                                search_kwargs={"k": 5}
                            ),
                            return_source_documents=True,
                            chain_type_kwargs={"prompt": PROMPT}
                        )
                        
                        # Get response
                        response = qa_chain.invoke({"query": query})
                        answer = response["result"]
                        
                        st.markdown(answer)
                        
                        # Show source documents
                        with st.expander("📖 View Source Documents"):
                            for i, doc in enumerate(response["source_documents"]):
                                st.markdown(f"**Source {i+1}:**")
                                st.text(doc.page_content[:300] + "...")
                                st.divider()
                        
                        # Add assistant response to chat history
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                        
                    except Exception as e:
                        error_msg = f"❌ Error generating response: {str(e)}"
                        st.error(error_msg)
                        st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
            
else:
    st.info("👈 Please upload and process a PDF document from the sidebar to start chatting!")
    st.markdown("""
    ### How to use this app:
    1. **Upload a PDF** in the sidebar
    2. Click **"Proceed to Vectorize"** to process the document
    3. Wait for the success message
    4. **Select documents** you want to chat with
    5. **Ask questions** about your documents!
    
    The AI will only answer based on the content of your uploaded documents.
    """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>Powered by LangChain, Gemini AI & ChromaDB</small>
</div>
""", unsafe_allow_html=True)