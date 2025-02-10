from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torchvision import transforms
from PIL import Image
import torch

dataset_split = []
train_set_augmented = []
valid_set_augmented = []
test_set_augmented = []

dataset = load_dataset("valhalla/emoji-dataset")

#Split data by text
for data in dataset["train"]:
    label = data["text"]
    if label.find("face") != -1:
        dataset_split.append(data)
        
#Split data into training, validation, and testing sets
train_set, temp_set = train_test_split(dataset_split, test_size=0.4)
test_set, valid_set = train_test_split(temp_set, test_size=0.5)

print(train_set)

transformations = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def augment_images(dataset):
    augmented_data = []
    for data in dataset:
        image = data["image"].convert("RGB")  # Convert to PIL Image
        for _ in range(5):  # Augment each image 5 times
            augmented_data.append(transformations(image))
    return augmented_data

# Apply augmentation
train_set_augmented = augment_images(train_set)
valid_set_augmented = augment_images(valid_set)
test_set_augmented = augment_images(test_set)

# Convert lists to tensors
train_set_augmented = torch.stack(train_set_augmented)
valid_set_augmented = torch.stack(valid_set_augmented)
test_set_augmented = torch.stack(test_set_augmented)

print(f"Original train set: {len(train_set)}, Augmented train set: {len(train_set_augmented)}")
print(f"Original valid set: {len(valid_set)}, Augmented valid set: {len(valid_set_augmented)}")
print(f"Original test set: {len(test_set)}, Augmented test set: {len(test_set_augmented)}")
    

        

        

        


