from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ---------------------------------------
# Module 8 : Greedy Decoding From Scratch
# ---------------------------------------

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

prompt = input("Enter Prompt: ")

# Convert to tensor
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

max_new_tokens = 20

print("\nGenerating...\n")

for step in range(max_new_tokens):

    with torch.no_grad():
        outputs = model(input_ids=input_ids)

    # Get logits of last token
    logits = outputs.logits[:, -1, :]

    # Greedy Selection
    next_token = torch.argmax(logits, dim=-1)

    # Append token
    input_ids = torch.cat(
        [input_ids, next_token.unsqueeze(-1)],
        dim=-1
    )

    # Stop if EOS token
    if next_token.item() == tokenizer.eos_token_id:
        break

# Decode entire sentence
generated_text = tokenizer.decode(
    input_ids[0],
    skip_special_tokens=True
)

print("Generated Text:\n")
print(generated_text)