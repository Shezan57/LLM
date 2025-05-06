import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize model architecture and tokenizer

model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)


# Load the fine-tuned state dictionary
state_dict = torch.load("data/deepseek-r1-finetuned-with-medical-data.pth", map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

model.eval()
