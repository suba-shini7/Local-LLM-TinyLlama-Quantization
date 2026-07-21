from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ---------------------------------------
# Module 13 : Dynamic Entropy Analysis
# ---------------------------------------

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

prompt = input("Enter Prompt: ")

input_ids = tokenizer(prompt, return_tensors="pt").input_ids

max_new_tokens = 15

print("\nGenerating with Entropy Analysis...\n")

for step in range(max_new_tokens):

    with torch.no_grad():
        outputs = model(input_ids=input_ids)

    # Logits of next token
    logits = outputs.logits[:, -1, :]

    # Convert to probabilities
    probabilities = torch.softmax(logits, dim=-1)

    # Entropy calculation
    entropy = -torch.sum(
        probabilities * torch.log(probabilities + 1e-10)
    )

    # Greedy decoding
    next_token = torch.argmax(probabilities, dim=-1)

    token = tokenizer.decode(next_token)

    print(f"Step {step+1}")
    print(f"Generated Token : {repr(token)}")
    print(f"Entropy         : {entropy.item():.4f}")
    print("-" * 40)

    # Append token
    input_ids = torch.cat(
        [input_ids, next_token.unsqueeze(-1)],
        dim=-1
    )

    if next_token.item() == tokenizer.eos_token_id:
        break

print("\nFinal Output\n")

print(tokenizer.decode(
    input_ids[0],
    skip_special_tokens=True
))