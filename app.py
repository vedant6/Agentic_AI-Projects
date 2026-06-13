import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

st.set_page_config(
    page_title="Vedant's RAG Pipeline",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
    <style>
    .main {background-color: #0f0f13;}
    .stTextInput input {background-color: #1a1a22; color: white;}
    .answer-box {
        background-color: #1a1a22;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #8B0000;
        color: white;
        margin-top: 10px;
    }
    .chunk-box {
        background-color: #1a2a1a;
        padding: 10px;
        border-radius: 8px;
        border-left: 3px solid #1dbd8c;
        color: #cccccc;
        margin: 5px 0;
        font-size: 12px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 RAG Pipeline — Vedant Aggarwal")
st.caption("Built with LangChain + ChromaDB + Groq LLaMA | TCS Interview Project")

with st.sidebar:
    st.header("📚 Knowledge Base")
    st.info("This RAG system answers questions from custom documents using vector search + LLM")
    
    st.subheader("How it works:")
    st.markdown("""
    1. 📄 Documents split into chunks
    2. 🔢 Chunks converted to vectors
    3. 💾 Stored in ChromaDB
    4. 🔍 Question matched to similar chunks
    5. 🤖 LLM answers from context
    """)
    
    st.subheader("Tech Stack:")
    st.markdown("""
    - **LangChain** — Pipeline framework
    - **ChromaDB** — Vector database
    - **HuggingFace** — Embeddings model
    - **Groq + LLaMA** — LLM backbone
    - **Streamlit** — UI framework
    """)

@st.cache_resource
def initialize_rag():
    documents = [
        Document(page_content="""TCS Digital Transformation Division focuses on 
        AI and ML solutions. The team works on RAG pipelines, LangChain agents, 
        and agentic workflows. Key technologies include Python, LangChain, 
        ChromaDB, and cloud platforms like AWS and Azure."""),
        
        Document(page_content="""Agentic AI systems at TCS use the ReAct pattern — 
        Reasoning and Acting in loops. Agents can call tools like search engines, 
        databases and APIs autonomously. Multi-agent systems coordinate multiple 
        specialized agents for complex enterprise tasks."""),
        
        Document(page_content="""RAG or Retrieval Augmented Generation improves 
        LLM accuracy by retrieving relevant documents before generating answers. 
        It uses vector databases like ChromaDB to store embeddings. 
        This prevents hallucination and keeps responses grounded in real data."""),
        
        Document(page_content="""Vedant Aggarwal is a Data Analyst with experience 
        in SQL, Power BI, ETL pipelines and data visualization. He is pursuing 
        Post Graduate Executive Certification in Data Science and AI from 
        IIT Roorkee. He has built RAG pipelines and LangChain agents."""),
        
        Document(page_content="""LangChain is an open source Python framework 
        that simplifies building LLM applications. It provides chains, agents, 
        memory, prompt templates and output parsers. LCEL uses pipe operator 
        to connect components in a clean readable flow."""),
        
        Document(page_content="""Prompt Engineering is the art of crafting inputs 
        to get best outputs from LLMs. Techniques include zero-shot, few-shot, 
        chain-of-thought and self-consistency prompting. ReAct pattern combines 
        reasoning and acting for agentic workflows."""),
    ]
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200, chunk_overlap=50
    )
    splits = splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings
    )
    
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile"
    )
    
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful AI assistant. Answer the question based on the context provided.
    Be concise and clear.
    
    Context: {context}
    
    Question: {question}
    
    Answer:
    """)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    rag_chain = (
        {"context": retriever | format_docs,
         "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 Ask a Question")
    
    question = st.text_input(
        "Type your question here:",
        placeholder="e.g. What is RAG? How do agents work? Tell me about Vedant..."
    )
    
    example_questions = [
        "What is RAG and how does it prevent hallucination?",
        "What technologies does TCS use for agentic AI?",
        "Tell me about Vedant's experience",
        "What is the ReAct pattern?",
        "How does LangChain work?",
    ]
    
    st.caption("Quick questions:")
    cols = st.columns(3)
    for i, eq in enumerate(example_questions[:3]):
        if cols[i].button(eq[:30] + "...", key=f"q{i}"):
            question = eq

with col2:
    st.subheader("📊 Pipeline Stats")
    st.metric("Documents", "6")
    st.metric("Chunks Created", "~12")
    st.metric("Embedding Model", "MiniLM-L6")
    st.metric("LLM", "LLaMA 3.3 70B")

if question:
    with st.spinner("🔍 Searching knowledge base and generating answer..."):
        try:
            rag_chain, retriever = initialize_rag()
            
            col_ans, col_chunks = st.columns([3, 2])
            
            with col_ans:
                st.subheader("🤖 Answer")
                answer = rag_chain.invoke(question)
                st.markdown(f'<div class="answer-box">{answer}</div>', 
                          unsafe_allow_html=True)
            
            with col_chunks:
                st.subheader("📄 Retrieved Chunks")
                st.caption("These chunks were used to answer your question")
                relevant_docs = retriever.invoke(question)
                for i, doc in enumerate(relevant_docs):
                    st.markdown(
                        f'<div class="chunk-box"><b>Chunk {i+1}:</b><br>{doc.page_content}</div>',
                        unsafe_allow_html=True
                    )
                    
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Make sure your GROQ_API_KEY is set in .env file")

st.markdown("---")
st.markdown("**Built by Vedant Aggarwal** | RAG Pipeline Project | LangChain + ChromaDB + Groq")