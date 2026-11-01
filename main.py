import os
import certifi
from dotenv import load_dotenv

# Gemini integration
from langchain_google_genai import ChatGoogleGenerativeAI

# Tavily search tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain_core.prompts import PromptTemplate

from langchain.tools import tool
import requests

load_dotenv()

from langchain.agents import create_react_agent, AgentExecutor

# ============================================
# LOAD ENV VARIABLES
# ============================================

os.environ["SSL_CERT_FILE"] = certifi.where()

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHERSTACK_API_KEY= os.getenv("WEATHERSTACK_API_KEY")

search_tool = TavilySearchResults(max_results=2)

@tool
def get_weather_data(city: str) -> str:
    """
    Fetch current weather information for a city.
    """

    url = (
        f"https://api.weatherstack.com/current?"
        f"access_key={WEATHERSTACK_API_KEY}&query={city}"
    )

    response = requests.get(url, timeout=15)

    data = response.json()

    if "current" not in data:
        return f"Could not fetch weather data for {city}: {data.get('error')}"

    return (
        f"Weather in {city}: "
        f"{data['current']['temperature']}°C, "
        f"{data['current']['weather_descriptions'][0]}. "
        f"Humidity: {data['current']['humidity']}%."
    )

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0,
    api_key=GOOGLE_API_KEY
)


#prompt
#prompt = hub.pull("hwchase17/react-multi-input-json")

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
6. If no tool is needed, directly provide Final Answer.
7. Never invent tool names.

Question: {input}

Thought:{agent_scratchpad}
"""

prompt = PromptTemplate.from_template(template)


#tool
tools = [search_tool, get_weather_data]


#create agent
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

#executor

agent_executor= AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5
)

#run
response = agent_executor.invoke({
    "input":(
        "Find the capital of India and find its current weather."
    )
})

print(response["output"])