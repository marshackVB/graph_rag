# LangGraph deployment examples on Databricks

This repository provides and example implementation of a LangGraph RAG application deployment on Databricks. The graph is wrapped by an MLflow [ChatAgent](https://docs.databricks.com/aws/en/generative-ai/agent-framework/author-agent) model (API documentation available [here](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.pyfunc.html#mlflow.pyfunc.ChatAgent)), enabling easy deployment on Mosaic AI model serving. The LangGraph state dictionary inherits from MLflow's [ChatAgentState](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.langchain.html#mlflow.langchain.chat_agent_langgraph.ChatAgentState), which ensures messages conform to MLflow's expected format. If calling tools within your LangGraph application (this example does not), [ChatAgentToolNode](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.langchain.html#mlflow.langchain.chat_agent_langgraph.ChatAgentToolNode) should wrap your tools instead of LangGraph's built in ToolNode. ChatAgent models return a [ChatAgentResponse](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.types.html#mlflow.types.agent.ChatAgentResponse), which is comprised of a list of [ChatAgentMessages](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.types.html#mlflow.types.agent.ChatAgentMessage). Although this application returns only messages, the format support custom outputs and attachments that could exist within your graph's state. See the ChatAgent API documentation linked above for more details.

The **chat_agent_deployment** notebook defines the ChatAgent and deploys the application to Databricks endpoint. The graph is defined in **graph.py** and inherits from other files in the repository.

The implementations is based on Databricks dbdemos advanced [RAG example](https://www.databricks.com/resources/demos/tutorials/data-science-and-ai/lakehouse-ai-deploy-your-llm-chatbot?itm_data=demo_center). See the folder, **03-advanced-app** and notebook, **02-Advanced-Chatbot-Chain**. This repo contains a LangGraph equivalent (roughly) of the demo's LangChain Expression Language (LCEL) implementation. Download the dbdemos notebooks and run the data preprocessing and vector index provisioning notebook, **01-PDF-Advanced-Data-Preparation**, also within the **03-advanced-app** folder. The vector search index created by this notebook is a requirement for the code in this repository.

You can install requirements within the notebook by running the commands below. Or, to prevent the need for continually reinstalling the dependecies, point to requirments.txt from within the cluster's Libraries tab, click install new, choose Workspace as the Library Source, and the dependencies will be installed at cluster provisioning time and will be available across all notebooks.
```
%sh pip install -r requirements.txt
%sh restart_python
```

You will need to update the resources names in the config.yaml file to match those in your Databricks Workspace. You can include any parameters in this file that would be helpful for iterating over to test different application configurations.

The workflow receives input messages and checks to see if historical messages (prior interactions between the user and model) exist. If they do, the model is prompted to rewrite the user's current question to include relevent context from prior interactions. The reframed question is then used to query the vector database. If there were no historical messages, the query is based on the user's precise question.
Finally, the context from the vector database, and either the users's question or model's reframed question format a prompt, and the final model call return the answer.

![langgraph workflow](img/graph.png)

### Useful info

 - This repo uses mlflow's [models from code](https://mlflow.org/docs/latest/model/models-from-code.html) approach to logging.
 - The [graph definitions](https://langchain-ai.github.io/langgraph/reference/graphs/#graph-definitions) in the LangGraph documentation are helpful for understanding the options for constructing workflows.
 - This project logs supporting scripts with the model using code_paths. There are some [limitations to be aware of](https://mlflow.org/docs/latest/model/dependencies.html#caveats-of-code-paths-option) that matter for how the project is structure. This project was originally formatted in a different way, but the directory structure could not be reflected within the MLflow model artifacts.
