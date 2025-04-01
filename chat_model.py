from typing import Optional, Any, Generator
import mlflow
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext)

from graph import load_graph

class GraphChatAgent(ChatAgent):
  def __init__(self):
    self.agent = load_graph()
    mlflow.langchain.autolog()

  def predict(self, 
              messages: list[ChatAgentMessage],
              context: Optional[ChatContext] = None,
              custom_inputs: Optional[dict[str, Any]] = None
              ) -> ChatAgentResponse:
    
    request = {"messages": self._convert_messages_to_dict(messages)}

    messages = []
    for event in self.agent.stream(request, stream_mode="updates"):
        for node_data in event.values():
            messages.extend(
                ChatAgentMessage(**msg) for msg in node_data.get("messages", [])
            )
    return ChatAgentResponse(messages=messages)
  

  def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:

        request = {"messages": self._convert_messages_to_dict(messages)}

        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                yield from (
                    ChatAgentChunk(**{"delta": msg}) for msg in node_data.get("messages", [])
                )

mlflow.models.set_model(GraphChatAgent())
