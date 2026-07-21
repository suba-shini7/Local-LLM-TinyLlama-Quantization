from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ---------------------------------------
# Module 7 : Softmax from Scratch
# ---------------------------------------

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# User input
prompt = input("Enter a prompt: ")

# Tokenize
inputs = tokenizer(prompt, return_tensors="pt")

# Forward pass
with torch.no_grad():
    outputs = model(**inputs)

# Get logits for next token
logits = outputs.logits[:, -1, :]

# ---------------------------------------
# Softmax From Scratch
# ---------------------------------------

# Step 1: Improve numerical stability
max_logit = torch.max(logits)

# Step 2: Exponential
exp_logits = torch.exp(logits - max_logit)

# Step 3: Sum of exponentials
sum_exp = torch.sum(exp_logits)

# Step 4: Probability
probabilities = exp_logits / sum_exp

# ---------------------------------------
# Top 10 Probabilities
# ---------------------------------------

k = 10

top_probs, top_indices = torch.topk(probabilities, k)

print("\nTop 10 Predicted Tokens")
print("=" * 75)
print(f"{'Rank':<5}{'Token':<20}{'Token ID':<12}{'Probability'}")
print("=" * 75)

for i in range(k):

    token_id = top_indices[0][i].item()

    token = tokenizer.decode([token_id])

    probability = top_probs[0][i].item()

    print(f"{i+1:<5}{repr(token):<20}{token_id:<12}{probability:.6f}")

print("=" * 75)

print("\nTotal Probability:", probabilities.sum().item())