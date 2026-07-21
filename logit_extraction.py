from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ---------------------------------------
# Module 6 : Logit Extraction
# ---------------------------------------

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Prompt
prompt = input("Enter a prompt: ")

# Tokenize
inputs = tokenizer(prompt, return_tensors="pt")

# Forward pass
with torch.no_grad():
    outputs = model(**inputs)

# Extract logits for the NEXT token
next_token_logits = outputs.logits[:, -1, :]

print("\nShape of Next Token Logits:")
print(next_token_logits.shape)

# Top-K predictions
k = 10
top_logits, top_indices = torch.topk(next_token_logits, k)

print("\nTop 10 Next Token Predictions")
print("=" * 65)
print(f"{'Rank':<5}{'Token':<20}{'Token ID':<12}{'Logit'}")
print("=" * 65)

for i in range(k):
    token_id = top_indices[0][i].item()
    token = tokenizer.decode([token_id])
    logit = top_logits[0][i].item()

    print(f"{i+1:<5}{repr(token):<20}{token_id:<12}{logit:.4f}")

print("=" * 65)