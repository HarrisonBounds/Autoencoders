from datasets import load_dataset

dataset_split = []
dataset = load_dataset("valhalla/emoji-dataset")

print(dataset)

for data in dataset["train"]:
    label = data["text"]
    if label.find("face") != -1:
        dataset_split.append(data)
        
print(len(dataset_split))

