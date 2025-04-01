from typing import List
from mlflow.langchain.chat_agent_langgraph import ChatAgentState


class GraphState(ChatAgentState):
  generated_question: List[dict[str,str]]
  context: List[str]