import re
text=input("Enter: ")
# text = "Hello! I am learning Python. It is easy & useful. Let's learn NLP!"
print("Original Text:")
print(text)
text = text.lower()

clean_text = re.sub(r'[^a-zA-Z0-9\s.!?]', '', text)

print("\nCleaned Text:")
print(clean_text)
sentences = re.split(r'[.!?]+', clean_text)

sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

print("\nSentence Tokens:")
print(sentences)

words = re.findall(r'\b\w+\b', clean_text)

print("\nWord Tokens:")
print(words)