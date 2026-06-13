# 🤖 Agentic AI Projects — Vedant Aggarwal

Three production-ready AI projects built with LangChain, LangGraph, and Groq LLaMA.
Developed as part of IBM RAG and Agentic AI Professional Certificate preparation.

---

## 🚀 Projects

### 1. RAG Pipeline (`app.py`)
**Retrieval Augmented Generation** — Prevents LLM hallucination by grounding 
responses in real documents.

**How it works:**
- Documents split into chunks using RecursiveCharacterTextSplitter
- Chunks converted to vectors using HuggingFace embeddings (all-MiniLM-L6-v2)
- Vectors stored in ChromaDB vector database
- User question matched to similar chunks via semantic search
- Groq LLaMA generates answer ONLY from retrieved chunks

**Tech Stack:** LangChain · ChromaDB · HuggingFace · Groq LLaMA · Streamlit

---

### 2. Agentic AI with ReAct Pattern (`agent_project.py`)
**Autonomous Agent** — Uses ReAct pattern (Reasoning + Acting) to 
autonomously decide which tools to use.

**How it works:**
- Agent receives a goal from user
- Reasons about which tool to use
- Acts by calling the tool
- Observes the result
- Repeats until goal achieved

**Available Tools:**
- 🧮 Calculator — Mathematical operations
- 🌤️ Weather — City weather information
- 📊 Data Analyzer — Statistical analysis
- 🔍 Company Search — Company information

**Tech Stack:** LangGraph · ReAct Pattern · Custom Tools · Groq LLaMA · Streamlit

---

### 3. Multi-Agent Orchestration System (`multi_agent.py`)
**4 Specialized Agents** coordinating autonomously to complete complex tasks.

**Agent Pipeline:**
- 🟡 **Orchestrator Agent** — Receives goal, plans and coordinates
- 🟢 **Researcher Agent** — Gathers and summarizes information
- 🔵 **Analyst Agent** — Analyzes data and finds insights
- 🟣 **Writer Agent** — Creates final professional report

**Tech Stack:** LangChain · Multi-Agent Pattern · Groq LLaMA · Streamlit

---

## ⚙️ Setup

### Prerequisites
- Python 3.11+
- Groq API Key (free at groq.com)

### Installation

### Environment Setup
Create `.env` file:

---

## 🧠 Key Concepts Demonstrated

| Concept | Where Used |
|---|---|
| RAG Pipeline | app.py |
| Vector Database | app.py (ChromaDB) |
| Embeddings | app.py (HuggingFace) |
| ReAct Pattern | agent_project.py |
| Tool Calling | agent_project.py |
| Function Calling | agent_project.py |
| Multi-Agent Orchestration | multi_agent.py |
| Agent Coordination | multi_agent.py |
| Prompt Engineering | All projects |
| LangChain LCEL | All projects |

---

## 📚 Tech Stack

- **LangChain** — LLM application framework
- **LangGraph** — Agent and multi-agent workflows
- **ChromaDB** — Vector database for semantic search
- **HuggingFace** — Embeddings model (all-MiniLM-L6-v2)
- **Groq + LLaMA 3.3 70B** — LLM backbone
- **Streamlit** — Web UI framework
- **Python** — Core language

---

## 👨‍💻 About

**Vedant Aggarwal**
- Data Analyst | AI Engineer
- MCA from GGSIPU
- Post Graduate Executive Certification in Data Science & AI — IIT Roorkee & Intellipaat
- Certified: Google Data Analytics · Microsoft Power BI · UC Davis Tableau

📧 vedant.aggarwal@outlook.com
🔗 linkedin.com/in/vedant-aggarwal
🌐 codingdojoacademy.com

---

*Built for TCS Agentic AI Interview — June 2026*
