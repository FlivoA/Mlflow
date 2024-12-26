import mlflow
import mlflow.pytorch
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Initialize model and tokenizer
model_name = "gpt2"  # Change to your model's path if using a different model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Log model to MLflow
mlflow.start_run()

# Log model
mlflow.pytorch.log_model(model, "chatbot_model")
mlflow.log_param("model_name", model_name)

# End the MLflow run
mlflow.end_run()

print("Model logged to MLflow successfully!")
