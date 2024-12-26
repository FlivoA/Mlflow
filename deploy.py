from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer from MLflow
import mlflow
import mlflow.pytorch

model_name = "chatbot_model"  # This should match the model logged in MLflow

# Load the model from MLflow
logged_model = mlflow.pytorch.load_model(f"runs:/{model_name}/chatbot_model")

tokenizer = AutoTokenizer.from_pretrained("gpt2")

app = FastAPI()

class MessageRequest(BaseModel):
    message: str

@app.post("/chat/")
async def chat(request: MessageRequest):
    # Tokenize the input message
    inputs = tokenizer.encode(request.message, return_tensors="pt")
    outputs = logged_model.generate(inputs, max_length=100, num_return_sequences=1)
    
    # Decode the output and return as response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
