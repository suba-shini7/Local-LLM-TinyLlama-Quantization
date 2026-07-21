from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ---------------------------------------
# Module 9 : Top-K Sampling
# ---------------------------------------

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

prompt = input("Enter Prompt: ")

input_ids = tokenizer(prompt, return_tensors="pt").input_ids

max_new_tokens = 20
k = 10

print("\nGenerating...\n")

for _ in range(max_new_tokens):

    with torch.no_grad():
        outputs = model(input_ids=input_ids)

    logits = outputs.logits[:, -1, :]

    # Convert logits to probabilities
    probabilities = torch.softmax(logits, dim=-1)

    # Keep only Top-K tokens
    top_probs, top_indices = torch.topk(probabilities, k)

    # Normalize probabilities
    top_probs = top_probs / torch.sum(top_probs)

    # Randomly choose one token
    sampled_index = torch.multinomial(top_probs, 1)

    next_token = top_indices.gather(-1, sampled_index)

    # Append token
    input_ids = torch.cat([input_ids, next_token], dim=-1)

    # Stop at EOS
    if next_token.item() == tokenizer.eos_token_id:
        break

generated_text = tokenizer.decode(
    input_ids[0],
    skip_special_tokens=True
)

print("\nGenerated Text:\n")
print(generated_text)