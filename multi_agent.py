import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

st.set_page_config(
    page_title="Multi-Agent System",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
    <style>
    .main {background-color: #0f0f13;}
    .agent-box {
        padding: 15px;
        border-radius: 10px;
        margin: 8px 0;
        color: white;
    }
    .orchestrator {border-left: 4px solid #FFD700; background-color: #1a1a00;}
    .researcher {border-left: 4px solid #1dbd8c; background-color: #001a0f;}
    .analyst {border-left: 4px solid #4A8FE7; background-color: #000f1a;}
    .writer {border-left: 4px solid #9B59B6; background-color: #0f001a;}
    </style>
""", unsafe_allow_html=True)

st.title("🤖 Multi-Agent Orchestration System")
st.caption("Built with LangChain + Groq | TCS Interview Project")

with st.sidebar:
    st.header("🏗️ System Architecture")
    st.markdown("""
    **4 Specialized Agents:**
    
    🟡 **Orchestrator Agent**
    Receives goal, plans and coordinates
    
    🟢 **Researcher Agent** 
    Gathers and researches information
    
    🔵 **Analyst Agent**
    Analyzes data and finds insights
    
    🟣 **Writer Agent**
    Creates final professional report
    """)
    st.subheader("Tech Stack:")
    st.markdown("""
    - **LangChain** — Agent framework
    - **Groq + LLaMA** — LLM for each agent
    - **Multi-agent pattern** — Orchestration
    - **Streamlit** — UI
    """)

def create_agent(system_prompt):
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile",
        temperature=0.3
    )
    def agent(task):
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=task)
        ]
        response = llm.invoke(messages)
        return response.content
    return agent

orchestrator = create_agent("""You are an Orchestrator Agent. 
Your job is to receive a goal and break it down into specific tasks 
for three specialized agents: Researcher, Analyst, and Writer.
Be concise and specific in task assignments.""")

researcher = create_agent("""You are a Research Agent specialized in 
gathering and summarizing information. Provide factual, relevant 
information about the given topic in a structured way.""")

analyst = create_agent("""You are an Analyst Agent specialized in 
analyzing information and finding insights, patterns and recommendations.
Provide data-driven insights and actionable recommendations.""")

writer = create_agent("""You are a Writer Agent specialized in creating 
clear, professional reports. Synthesize information from research and 
analysis into a well-structured final report.""")

st.subheader("🎯 Give the System a Goal")

examples = [
    "Analyze the impact of AI on the Indian IT job market",
    "Research and analyze TCS as a company to work for",
    "Analyze career prospects for Data Analysts in 2026",
]

st.caption("Example goals:")
cols = st.columns(3)
selected = ""
for i, ex in enumerate(examples):
    if cols[i].button(ex[:25]+"...", key=f"ex{i}"):
        selected = ex

goal = st.text_input("Enter your goal:", 
    value=selected,
    placeholder="e.g. Analyze the future of AI in enterprise software")

if goal:
    st.markdown("---")
    st.subheader("🔄 Multi-Agent Pipeline Running...")
    
    # Step 1 - Orchestrator
    with st.spinner("🟡 Orchestrator planning..."):
        orchestrator_output = orchestrator(
            f"Goal: {goal}\nBreak this into tasks for Researcher, Analyst and Writer agents."
        )
    
    st.markdown(f'<div class="agent-box orchestrator"><b>🟡 ORCHESTRATOR AGENT</b><br><br>{orchestrator_output}</div>', 
               unsafe_allow_html=True)
    
    # Step 2 - Researcher
    with st.spinner("🟢 Researcher gathering information..."):
        research_output = researcher(
            f"Research this topic thoroughly: {goal}"
        )
    
    st.markdown(f'<div class="agent-box researcher"><b>🟢 RESEARCHER AGENT</b><br><br>{research_output}</div>', 
               unsafe_allow_html=True)
    
    # Step 3 - Analyst
    with st.spinner("🔵 Analyst finding insights..."):
        analyst_output = analyst(
            f"Analyze this research and provide insights:\n{research_output[:500]}\n\nOriginal goal: {goal}"
        )
    
    st.markdown(f'<div class="agent-box analyst"><b>🔵 ANALYST AGENT</b><br><br>{analyst_output}</div>', 
               unsafe_allow_html=True)
    
    # Step 4 - Writer
    with st.spinner("🟣 Writer creating final report..."):
        writer_output = writer(
            f"""Create a final report for: {goal}
            
Research findings: {research_output[:400]}
Analysis insights: {analyst_output[:400]}

Write a professional, structured final report."""
        )
    
    st.markdown(f'<div class="agent-box writer"><b>🟣 WRITER AGENT — FINAL REPORT</b><br><br>{writer_output}</div>', 
               unsafe_allow_html=True)
    
    st.success("✅ Multi-Agent Pipeline Complete!")

st.markdown("---")
st.markdown("**Built by Vedant Aggarwal** | Multi-Agent Orchestration | LangChain + Groq")