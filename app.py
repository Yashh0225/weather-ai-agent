
import os
import certifi
import requests
import streamlit as st

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor


# ============================================
# 1. CONFIGURATION
# ============================================

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")


# ============================================
# 2. STREAMLIT PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="AI Agent Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI Agent Assistant")
st.caption("Powered by Gemini, LangChain, Tavily and Weatherstack")


# ============================================
# 3. VALIDATE API KEYS
# ============================================

if not GOOGLE_API_KEY or not TAVILY_API_KEY or not WEATHERSTACK_API_KEY:
    st.error(
        "Missing API keys. Please check GOOGLE_API_KEY, "
        "TAVILY_API_KEY and WEATHERSTACK_API_KEY in your .env file."
    )
    st.stop()


# ============================================
# 4. DEFINE WEATHER TOOL
# ============================================

@tool
def get_weather_data(city: str) -> str:
    """
    Fetch current weather information for a city.
    """

    url = "https://api.weatherstack.com/current"

    params = {
        "access_key": WEATHERSTACK_API_KEY,
        "query": city
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()

        data = response.json()

        if "current" not in data:
            return (
                f"Could not fetch weather data for {city}: "
                f"{data.get('error', 'Unknown API error')}"
            )

        current = data["current"]

        return (
            f"Weather in {city}: "
            f"{current['temperature']}°C, "
            f"{current['weather_descriptions'][0]}. "
            f"Humidity: {current['humidity']}%."
        )

    except requests.RequestException as e:
        return f"Weather API request failed: {str(e)}"


# ============================================
# 5. CREATE AGENT
# ============================================

@st.cache_resource
def create_agent():

    # LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        temperature=0,
        api_key=GOOGLE_API_KEY
    )

    # Tools
    search_tool = TavilySearchResults(max_results=2)

    tools = [search_tool, get_weather_data]

    # Custom ReAct prompt
    template = """
You are a helpful AI assistant that can use tools.

You have access to these tools:

{tools}

You MUST follow this exact format:

Question: the user's question

Thought: explain what you need to do.

Action: one of [{tool_names}]
Action Input: the input for the selected tool

Observation: the result returned by the tool.

Repeat Thought, Action, Action Input, and Observation
if more tools are required.

When you have enough information, use:

Thought: I now know the final answer.
Final Answer: your final response.

STRICT RULES:
1. Never use JSON for actions.
2. Never use Markdown code blocks.
3. Action must contain only the exact tool name.
4. Action Input must contain the tool input.
5. Never combine Action and Action Input on one line.
6. If no tool is needed, directly provide the Final Answer.
7. Never invent tool names.

Question: {input}

Thought:{agent_scratchpad}
"""

    prompt = PromptTemplate.from_template(template)

    # Agent
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    # Agent Executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5
    )

    return agent_executor


# ============================================
# 6. INITIALIZE AGENT
# ============================================

try:
    agent_executor = create_agent()
except Exception as e:
    st.error(f"Could not initialize agent: {str(e)}")
    st.stop()


# ============================================
# 7. CHAT HISTORY
# ============================================

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I'm your AI agent. Ask me anything!"
        }
    ]


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================
# 8. HANDLE USER INPUT
# ============================================

user_input = st.chat_input("Ask your AI agent something...")

if user_input:

    # Display user message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate agent response
    with st.chat_message("assistant"):

        with st.spinner("Agent is thinking and using tools..."):

            try:
                response = agent_executor.invoke({
                    "input": user_input
                })

                answer = response["output"]

            except Exception as e:
                answer = f"Sorry, something went wrong: {str(e)}"

        st.markdown(answer)

    # Save assistant response
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
