from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ---------------------------------------
# Module 16 : Complete Analytical Pipeline
# ---------------------------------------

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    output_hidden_states=True,
    output_attentions=True
)

prompt = input("Enter Prompt: ")

input_ids = tokenizer(prompt, return_tensors="pt").input_ids

max_new_tokens = 10

print("\nStarting Analysis Pipeline...\n")

for step in range(max_new_tokens):

    with torch.no_grad():

        outputs = model(input_ids=input_ids)

    # -----------------------------
    # Logits
    # -----------------------------
    logits = outputs.logits[:, -1, :]

    probabilities = torch.softmax(logits, dim=-1)

    entropy = -torch.sum(
        probabilities * torch.log(probabilities + 1e-10)
    )

    # -----------------------------
    # Greedy Prediction
    # -----------------------------
    next_token = torch.argmax(probabilities, dim=-1)

    token = tokenizer.decode(next_token)

    token_probability = probabilities[0, next_token.item()].item()

    token_logit = logits[0, next_token.item()].item()

    # -----------------------------
    # Hidden States
    # -----------------------------
    hidden_layers = outputs.hidden_states

    last_hidden = hidden_layers[-1]

    hidden_shape = last_hidden.shape

    # -----------------------------
    # Attention
    # -----------------------------
    attention_layers = outputs.attentions

    attention_shape = attention_layers[-1].shape

    # -----------------------------
    # Print Analysis
    # -----------------------------
    print("=" * 70)

    print(f"Generation Step : {step+1}")

    print(f"Generated Token : {repr(token)}")

    print(f"Token ID        : {next_token.item()}")

    print(f"Logit Score     : {token_logit:.4f}")

    print(f"Probability     : {token_probability:.6f}")

    print(f"Entropy         : {entropy.item():.4f}")

    print(f"Hidden Shape    : {hidden_shape}")

    print(f"Attention Shape : {attention_shape}")

    # -----------------------------
    # Append Token
    # -----------------------------
    input_ids = torch.cat(
        [input_ids, next_token.unsqueeze(-1)],
        dim=-1
    )

    if next_token.item() == tokenizer.eos_token_id:
        break

print("\nFinal Output:\n")

print(tokenizer.decode(
    input_ids[0],
    skip_special_tokens=True
))
