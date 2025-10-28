from datasets import load_dataset

print("Start load")
dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
print("Loaded")