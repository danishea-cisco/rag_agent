from langchain.agents import create_agent
from rag.indexing import retrieve_context
from daniel_agent.utils.model import chat_model
from langgraph.checkpoint.memory import InMemorySaver
from daniel_agent.utils.prompts import DANIEL_AGENT_PROMPT

tools = [retrieve_context]

def get_agent():
    agent = create_agent(
        chat_model,
        tools,
        system_prompt=DANIEL_AGENT_PROMPT,
        checkpointer=InMemorySaver(),
    )
    return agent