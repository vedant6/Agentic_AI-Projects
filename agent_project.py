import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langgraph.prebuilt import create_react_agent

load_dotenv()

st.set_page_config(
    page_title="Agentic AI — Vedant",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
    <style>
    .main {background-color: #0f0f13;}
    .tool-box {
        background-color: #1a1a22;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #1dbd8c;
        color: white;
        margin: 8px 0;
    }
    .answer-box {
        background-color: #1a1a22;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #8B0000;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 Agentic AI — ReAct Pattern")
st.caption("Built with LangChain Agents + Groq LLaMA | TCS Interview Project")

with st.sidebar:
    st.header("🧠 How Agents Work")
    st.markdown("""
    **ReAct Pattern:**
    1. 🤔 **Reason** — Think what to do
    2. 🔧 **Act** — Use a tool
    3. 👀 **Observe** — Check result
    4. 🔄 **Repeat** — Until goal achieved
    """)
    st.subheader("Available Tools:")
    st.markdown("""
    - 🧮 **Calculator** — Math operations
    - 🌤️ **Weather** — City weather info
    - 📊 **Data Analyzer** — Analyze numbers
    - 🔍 **Company Search** — Company info
    """)
    st.subheader("Tech Stack:")
    st.markdown("""
    - **LangGraph** — Agent framework
    - **ReAct Pattern** — Reasoning loop
    - **Groq + LLaMA** — LLM backbone
    - **Custom Tools** — Function calling
    """)

@tool
def calculator(expression: str) -> str:
    """Useful for mathematical calculations. Input should be a math expression like '25 * 4' or '100 / 5'"""
    try:
        result = eval(expression)
        return f"Calculator result: {expression} = {result}"
    except:
        return "Invalid mathematical expression"

@tool
def get_weather(city: str) -> str:
    """Get weather information for a city. Input should be city name."""
    weather_data = {
        "delhi": "Delhi: 42°C, Sunny, Humidity 45%",
        "mumbai": "Mumbai: 32°C, Humid, Humidity 85%",
        "bangalore": "Bangalore: 28°C, Partly Cloudy, Humidity 60%",
        "noida": "Noida: 41°C, Sunny, Humidity 40%",
        "gurgaon": "Gurgaon: 41°C, Hot, Humidity 42%",
        "jaipur": "Jaipur: 44°C, Very Hot, Humidity 30%",
    }
    return weather_data.get(city.lower(), f"{city}: 35°C, Clear Sky, Humidity 50%")

@tool
def analyze_data(numbers: str) -> str:
    """Analyze a list of numbers. Input should be comma separated numbers like '10,20,30,40,50'"""
    try:
        nums = [float(x.strip()) for x in numbers.split(',')]
        avg = sum(nums) / len(nums)
        return f"Count: {len(nums)}, Sum: {sum(nums)}, Average: {avg:.2f}, Max: {max(nums)}, Min: {min(nums)}"
    except:
        return "Please provide valid comma-separated numbers"

@tool
def search_company(company: str) -> str:
    """Search information about a company. Input should be company name."""
    companies = {
        "tcs": "TCS: India's largest IT company. Revenue $25B+. Focuses on AI, Cloud, Digital transformation.",
        "concentrix": "Concentrix: Fortune 500 global technology company. Customer experience and business transformation.",
        "google": "Google: Multinational tech company. AI focus with Gemini and DeepMind.",
        "microsoft": "Microsoft: Global tech giant. AI focus with Copilot and OpenAI partnership.",
    }
    return companies.get(company.lower(), f"{company}: Leading technology company with global presence.")

@st.cache_resource
def get_agent():
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile",
        temperature=0
    )
    tools = [calculator, get_weather, analyze_data, search_company]
    agent = create_react_agent(llm, tools)
    return agent

st.subheader("💬 Give the Agent a Task")

example_tasks = [
    "What is the weather in Delhi and Jaipur?",
    "Calculate 15% of 85000 salary",
    "Analyze this data: 45000,52000,48000,61000,55000",
    "Tell me about TCS company",
    "What is 2500 * 12 and what is the weather in Noida?",
]

st.caption("Example tasks:")
cols = st.columns(3)
selected_task = ""
for i, task in enumerate(example_tasks[:3]):
    if cols[i].button(task[:25]+"...", key=f"t{i}"):
        selected_task = task

task = st.text_input("Or type your own task:",
    value=selected_task,
    placeholder="e.g. What is the weather in Mumbai and calculate 20% of 50000")

if task:
    with st.spinner("🤔 Agent is thinking and acting..."):
        try:
            agent = get_agent()
            result = agent.invoke({"messages": [{"role": "user", "content": task}]})
            final_answer = result["messages"][-1].content

            col1, col2 = st.columns([3, 2])

            with col1:
                st.subheader("✅ Final Answer")
                st.markdown(
                    f'<div class="answer-box">{final_answer}</div>',
                    unsafe_allow_html=True
                )

            with col2:
                st.subheader("🔧 Tools Used")
                st.markdown(
                    '<div class="tool-box">Agent used ReAct pattern:<br>Reason → Act → Observe → Repeat<br><br>All steps processed autonomously!</div>',
                    unsafe_allow_html=True
                )

        except Exception as e:
            st.error(f"Error: {str(e)}")

st.markdown("---")
st.markdown("**Built by Vedant Aggarwal** | Agentic AI Project | LangGraph ReAct + Groq")