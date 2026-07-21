from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import matplotlib.pyplot as plt

# ---------------------------------------
# Module 14 : Logit Distribution Visualization
# ---------------------------------------

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

prompt = input("Enter Prompt: ")

inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# Next-token logits
logits = outputs.logits[:, -1, :]

# Convert logits into probabilities
probabilities = torch.softmax(logits, dim=-1)

# Top 10 predictions
k = 10

top_probs, top_indices = torch.topk(probabilities, k)

tokens = [
    tokenizer.decode([idx]).strip()
    for idx in top_indices[0]
]

values = top_probs[0].tolist()

# Plot
plt.figure(figsize=(10,5))
plt.bar(tokens, values)

plt.title("Top-10 Next Token Probability Distribution")
plt.xlabel("Tokens")
plt.ylabel("Probability")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()