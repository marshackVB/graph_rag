import mlflow
from langchain_core.runnables import RunnableGenerator, RunnableLambda
from graph import load_graph
from nodes.utils import graph_state_to_chat_type


def load_chain():
  app = load_graph()
  chain = app | RunnableLambda(graph_state_to_chat_type)
  return chain

mlflow.models.set_model(load_chain())
