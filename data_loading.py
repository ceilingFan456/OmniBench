from datasets import load_dataset, Audio

dataset = load_dataset("m-a-p/OmniBench")
dataset = dataset.cast_column("audio", Audio(decode=False))  # <- key line

# check on the data samples
print(dataset)
print(dataset['train'][0])

# similar for OmniInstruct
dataset = load_dataset("m-a-p/OmniInstruct_v1")

# check on the data samples
print(dataset)
print(dataset['train'][0])