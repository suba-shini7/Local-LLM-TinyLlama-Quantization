from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ---------------------------------------
# Module 10 : Top-P (Nucleus Sampling)
# ---------------------------------------

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

prompt = input("Enter Prompt: ")

input_ids = tokenizer(prompt, return_tensors="pt").input_ids

max_new_tokens = 20
top_p = 0.90

print("\nGenerating...\n")

for _ in range(max_new_tokens):

    with torch.no_grad():
        outputs = model(input_ids=input_ids)

    logits = outputs.logits[:, -1, :]

    probabilities = torch.softmax(logits, dim=-1)

    # Sort probabilities
    sorted_probs, sorted_indices = torch.sort(
        probabilities,
        descending=True
    )

    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Keep only probabilities within top_p
    nucleus_mask = cumulative_probs <= top_p

    # Always keep the highest probability token
    nucleus_mask[..., 0] = True

    filtered_probs = sorted_probs[nucleus_mask]
    filtered_indices = sorted_indices[nucleus_mask]

    # Normalize
    filtered_probs = filtered_probs / filtered_probs.sum()

    # Sample one token
    sampled = torch.multinomial(filtered_probs, 1)

    next_token = filtered_indices[sampled]

    # Append token
    input_ids = torch.cat(
        [input_ids, next_token.unsqueeze(0)],
        dim=-1
    )

    if next_token.item() == tokenizer.eos_token_id:
        break

generated_text = tokenizer.decode(
    input_ids[0],
    skip_special_tokens=True
)

print("\nGenerated Text:\n")
print(generated_text)