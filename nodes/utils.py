from functools import partial
from typing import List, Dict, Iterator
from mlflow.types.llm import ChatCompletionRequest, ChatCompletionResponse, ChatChoice, ChatMessage
from langchain_core.messages import MessageLikeRepresentation
from langgraph.graph import StateGraph
from dataclasses import asdict


def format_generation(role: str, generation) -> Dict[str, str]:
  """
  Reformat Chat model response to a list of dictionaries. This function
  is called within the graph's nodes to ensure a consistent chat
  format is saved in the graph's state
  """
  return [{"role": role, "content": generation.content}]


format_generation_user = partial(format_generation, "user")
format_generation_assistant = partial(format_generation, "assistant")


def choose_prompt_question(state: StateGraph) -> List[Dict[str, str]]:
  """
  Determine if the users's original question or the model's
  reformulated question should be used to query the vector index
  and generate the model's final answer.
  """
  if 'generated_question' in state:
    question = state['generated_question']
  else:
    question = [state['messages'][-1]]

  return question
