"""
This module implements a weather information agent using LangGraph and the Tavily search tool.
It can answer questions about current weather conditions for various locations.
"""

import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Sequence, List, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Ensure API keys are set
if not OPENAI_API_KEY or not TAVILY_API_KEY:
    raise ValueError("Please set OPENAI_API_KEY and TAVILY_API_KEY in your .env file")

class AgentState(TypedDict):
    """
    Represents the state of the agent during a conversation.

    Attributes:
        messages (Sequence[BaseMessage]): The messages in the conversation.
        current_weather (str): The current weather information.
        search_attempted (bool): Whether a weather search has been attempted.
    """
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    current_weather: Annotated[str, "The current weather information"]
    search_attempted: Annotated[bool, "Whether a weather search has been attempted"]

# Initialize the Tavily search tool
tavily_tool = TavilySearchResults(max_results=2)

# Set up the model
model = ChatOpenAI()

# Define the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant that answers questions about the weather. Use the provided weather information to answer questions accurately."),
    MessagesPlaceholder(variable_name="messages"),
    ("system", "Current weather information: {current_weather}"),
    ("human", "{input}"),
])

def agent(state: AgentState) -> Dict[str, Any]:
    """
    Processes the current state and generates a response using the AI model.

    Args:
        state (AgentState): The current state of the agent.

    Returns:
        Dict[str, Any]: Updated state with the new response.
    """
    messages = state["messages"]
    current_weather = state.get("current_weather", "No weather information available.")

    last_message = messages[-1]

    response = model.invoke(
        prompt.format_messages(
            messages=messages[:-1],
            current_weather=current_weather,
            input=last_message.content
        )
    )

    return {"messages": messages + [response], "current_weather": current_weather, "search_attempted": state["search_attempted"]}

def should_use_tool(state: AgentState) -> str:
    """
    Determines whether to use the weather search tool based on the current state.

    Args:
        state (AgentState): The current state of the agent.

    Returns:
        str: "search_weather" if a search should be performed, "end" otherwise.
    """
    if not state["search_attempted"]:
        return "search_weather"
    return "end"

def search_weather(state: AgentState) -> AgentState:
    """
    Performs a weather search using the Tavily tool.

    Args:
        state (AgentState): The current state of the agent.

    Returns:
        AgentState: Updated state with the weather information.
    """
    messages = state["messages"]
    last_message = messages[-1]

    search_results = tavily_tool.invoke(f"current weather {last_message.content}")

    if search_results and len(search_results) > 0:
        weather_info = search_results[0].get('content', "Unable to fetch weather information.")
    else:
        weather_info = "Unable to fetch weather information."

    return {"messages": messages, "current_weather": weather_info, "search_attempted": True}

# Create the graph
workflow = StateGraph(AgentState)

# Define the nodes
workflow.add_node("agent", agent)
workflow.add_node("search_weather", search_weather)

# Define the edges
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_use_tool,
    {
        "search_weather": "search_weather",
        "end": END
    }
)
workflow.add_edge("search_weather", "agent")

# Set the finish point
workflow.set_finish_point("agent")

# Compile the graph
app = workflow.compile()

def run_agent(query: str) -> str:
    """
    Runs the agent with a given query.

    Args:
        query (str): The user's question about the weather.

    Returns:
        str: The agent's response to the query.
    """
    inputs = {"messages": [HumanMessage(content=query)], "current_weather": "", "search_attempted": False}
    result = app.invoke(inputs)
    return result["messages"][-1].content

if __name__ == "__main__":
    query = "What is the weather in Sunnyvale, CA?"
    response = run_agent(query)
    print(f"User: {query}")
    print(f"Agent: {response}")