from datasets import load_dataset
from sklearn.model_selection import train_test_split

dataset_split = []
dataset = load_dataset("valhalla/emoji-dataset")

#Split data by text
for data in dataset["train"]:
    label = data["text"]
    if label.find("face") != -1:
        dataset_split.append(data)
        
#Split data into training, validation, and testing sets
train_set, temp_set = train_test_split(dataset_split, test_size=0.4)
test_set, valid_set = train_test_split(temp_set, test_size=0.5)


        

        

        


