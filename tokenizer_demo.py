from transformers import AutoTokenizer

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

text = "Hello, I am learning Large Language Models."

tokens = tokenizer(text)

print("Original Text:")
print(text)

print("\nToken IDs:")
print(tokens["input_ids"])

print("\nAttention Mask:")
print(tokens["attention_mask"])

decoded = tokenizer.decode(tokens["input_ids"])

print("\nDecoded Text:")
print(decoded)