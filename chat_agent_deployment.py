# Databricks notebook source
# MAGIC %md ##ChatAgent deployment 
# MAGIC
# MAGIC See the documentation for [ChatAgent](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.pyfunc.html#mlflow.pyfunc.ChatAgent) as well as an [additional examples](https://docs.databricks.com/aws/en/generative-ai/agent-framework/author-agent#chatagent-examples).
# MAGIC

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# MAGIC %md #### Define MLflow ChatAgent model

# COMMAND ----------

# MAGIC %%writefile chat_model.py
# MAGIC from typing import Optional, Any, Generator
# MAGIC import mlflow
# MAGIC from mlflow.pyfunc import ChatAgent
# MAGIC from mlflow.types.agent import (
# MAGIC     ChatAgentChunk,
# MAGIC     ChatAgentMessage,
# MAGIC     ChatAgentResponse,
# MAGIC     ChatContext)
# MAGIC
# MAGIC from graph import load_graph
# MAGIC
# MAGIC class GraphChatAgent(ChatAgent):
# MAGIC   def __init__(self):
# MAGIC     self.agent = load_graph()
# MAGIC     mlflow.langchain.autolog()
# MAGIC
# MAGIC   def predict(self, 
# MAGIC               messages: list[ChatAgentMessage],
# MAGIC               context: Optional[ChatContext] = None,
# MAGIC               custom_inputs: Optional[dict[str, Any]] = None
# MAGIC               ) -> ChatAgentResponse:
# MAGIC     
# MAGIC     request = {"messages": self._convert_messages_to_dict(messages)}
# MAGIC
# MAGIC     messages = []
# MAGIC     for event in self.agent.stream(request, stream_mode="updates"):
# MAGIC         for node_data in event.values():
# MAGIC             messages.extend(
# MAGIC                 ChatAgentMessage(**msg) for msg in node_data.get("messages", [])
# MAGIC             )
# MAGIC     return ChatAgentResponse(messages=messages)
# MAGIC   
# MAGIC
# MAGIC   def predict_stream(
# MAGIC         self,
# MAGIC         messages: list[ChatAgentMessage],
# MAGIC         context: Optional[ChatContext] = None,
# MAGIC         custom_inputs: Optional[dict[str, Any]] = None,
# MAGIC     ) -> Generator[ChatAgentChunk, None, None]:
# MAGIC
# MAGIC         request = {"messages": self._convert_messages_to_dict(messages)}
# MAGIC
# MAGIC         for event in self.agent.stream(request, stream_mode="updates"):
# MAGIC             for node_data in event.values():
# MAGIC                 yield from (
# MAGIC                     ChatAgentChunk(**{"delta": msg}) for msg in node_data.get("messages", [])
# MAGIC                 )
# MAGIC
# MAGIC mlflow.models.set_model(GraphChatAgent())

# COMMAND ----------

# MAGIC %md View graph

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from IPython.display import Image
import mlflow
from chat_model import GraphChatAgent

# COMMAND ----------

AGENT = GraphChatAgent()
mlflow.models.set_model(AGENT)

display(Image(AGENT.agent.get_graph().draw_mermaid_png(output_file_path="graph.png")))

# COMMAND ----------

# MAGIC %md Test single question response

# COMMAND ----------

response = AGENT.predict({"messages": [{"role": "user", "content": "What is Apache Spark?"}]})

# COMMAND ----------

# MAGIC %md Test follow up with message history

# COMMAND ----------

history = response.model_dump()['messages']
follow_up = {"messages": history + [{"role": "user", "content": "Does is support streaming?"}]}
follow_up_response = AGENT.predict(follow_up)

# COMMAND ----------

follow_up_response

# COMMAND ----------

# MAGIC %md Test streaming prediction

# COMMAND ----------

for event in AGENT.predict_stream(follow_up):
  print(f"{event}\n")

# COMMAND ----------

# MAGIC %md #### Log model to MLflow

# COMMAND ----------

import mlflow
from mlflow.models.resources import DatabricksServingEndpoint, DatabricksVectorSearchIndex
from data.input_examples import input_example, input_examples
from config_utils import load_config

retriever_config = load_config("retriever")
retriever_schema = retriever_config['schema']
vector_search_index_name = retriever_config['vector_search_index']

model_config = load_config("model")
model_name = model_config.get('name')

mlflow_config = load_config("mlflow")
experiment_location = mlflow_config["experiment_location"]
mlflow.set_experiment(experiment_location)
uc_model = mlflow_config["uc_model"]

mlflow.models.set_retriever_schema(
    primary_key=retriever_schema.get("primary_key"),
    text_column=retriever_schema.get("chunk_text"),
    doc_uri=retriever_schema.get("document_uri")  
)

with mlflow.start_run(run_name="graph_rag_pyfunc"):

  model_info = mlflow.pyfunc.log_model(
                  python_model = "chat_model.py",
                  streamable=True,
                  model_config="config.yaml",
                  artifact_path="graph",
                  input_example=input_example,
                  code_paths = [
                    'nodes',
                    'prompts',
                    'resources',
                    'state.py',
                    'graph.py',
                    'config_utils.py'
                    ],
                  resources = [
                    DatabricksVectorSearchIndex(index_name=vector_search_index_name),
                    DatabricksServingEndpoint(endpoint_name=model_name)
                    ],
                  pip_requirements = "requirements.txt"
               )
  
  mlflow.log_artifact("graph.png")

  model_uri = model_info.model_uri

  loaded_app = mlflow.pyfunc.load_model(model_uri)
  for example in input_examples:
    loaded_app.predict(example)
  
print(model_uri)

# COMMAND ----------

# MAGIC %md #### Validate model inference

# COMMAND ----------

from mlflow.models import validate_serving_input
from mlflow.models.utils import load_serving_example

serving_payload = load_serving_example(model_uri)

validate_serving_input(model_uri, serving_payload)

# COMMAND ----------

# MAGIC %md #### Load mlflow model and test inference types.

# COMMAND ----------

loaded_model = mlflow.pyfunc.load_model(model_uri)

# COMMAND ----------

# MAGIC %md Invoke

# COMMAND ----------

from pprint import pprint

predictions_invoke = loaded_model.predict(eval(serving_payload))
pprint(predictions_invoke)

# COMMAND ----------

# MAGIC %md Stream

# COMMAND ----------

streaming_results = []
for event in loaded_model.predict_stream(eval(serving_payload)):
  streaming_results.append(event)

# COMMAND ----------

# MAGIC %md ###Log as UC Model

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

model_info = mlflow.register_model(model_uri, 
                                   name = uc_model,
                                   tags={"model_type": "ChatAgent",
                                         "streaming": True})

# COMMAND ----------

# MAGIC %md ### Deploy as Agent

# COMMAND ----------

from databricks.agents import deploy
from config_utils import load_config

mlflow_config = load_config("mlflow")
uc_model = mlflow_config.get('uc_model')

deployment_info = deploy(model_name=uc_model, 
                         model_version=model_info.version)

# COMMAND ----------

# MAGIC %md ### Query the endpoint

# COMMAND ----------

from pprint import pprint 
from mlflow.deployments import get_deploy_client

deploy_client = get_deploy_client("databricks")
endpoint_name = f"agents_{uc_model.replace('.', '-')}"

# COMMAND ----------

# MAGIC %md Batch

# COMMAND ----------

deploy_client.predict(endpoint=deployment_info.endpoint_name, inputs=eval(serving_payload))

# COMMAND ----------

# MAGIC %md Stream

# COMMAND ----------

for event in deploy_client.predict_stream(endpoint=deployment_info.endpoint_name, inputs=eval(serving_payload)):
  print(event)
