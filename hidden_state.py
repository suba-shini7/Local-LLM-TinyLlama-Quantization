from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ---------------------------------------
# Module 11 : Hidden State Extraction
# ---------------------------------------

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    output_hidden_states=True
)

prompt = input("Enter Prompt: ")

inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():

    outputs = model(**inputs)

hidden_states = outputs.hidden_states

print("\nTotal Hidden Layers:", len(hidden_states))

print("-" * 60)

for i, layer in enumerate(hidden_states):

    print(f"Layer {i}")

    print("Shape:", layer.shape)

    print("-" * 60)
    