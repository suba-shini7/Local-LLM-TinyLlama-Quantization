from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ---------------------------------------
# Module 12 : Attention Weight Extraction
# ---------------------------------------

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    output_attentions=True
)

prompt = input("Enter Prompt: ")

inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

attentions = outputs.attentions

print("\nTotal Transformer Layers :", len(attentions))
print("=" * 60)

for layer_num, attention in enumerate(attentions):

    print(f"Layer {layer_num}")

    print("Attention Shape :", attention.shape)

    print("-" * 60)