import mlflow


def load_config(config_type, config_file='config.yaml'):
  """
  A helper function to access different configurations
  from the mlflow config file.
  """
  config = mlflow.models.ModelConfig(development_config=config_file)

  config_mapping = {"model": config.get("model"),
                    "langgraph": config.get("langgraph"),
                    "retriever": config.get("retriever"),
                    "mlflow": config.get("mlflow")}
  
  config_types = config_mapping.keys()
  
  if config_type in config_types:    
    return config_mapping.get(config_type)
  
  else:
    raise Exception(
      f"The config type, {config_type}, is not among the supported types, ({(", ").join(config_types)})."
    )