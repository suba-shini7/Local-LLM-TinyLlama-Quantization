from transformers import AutoTokenizer

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

print("Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Tokenizer downloaded!")