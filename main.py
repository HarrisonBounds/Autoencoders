from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torchvision import transforms
import torch
from model import AE
import matplotlib.pyplot as plt


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

transformations = transforms.Compose([
    transforms.Resize(64), 
    transforms.RandomHorizontalFlip(),
    #transforms.RandomRotation(30),
    #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

train_loader = torch.utils.data.DataLoader(dataset = train_set_augmented,
                                     batch_size = 32,
                                     shuffle = True)

model = AE(input_width=64, input_height=64, num_channels=3)
loss_func = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr = 1e-1,
                             weight_decay = 1e-8)

num_epochs = 20
losses = []
inputs = []
outputs = []
for epoch in range(num_epochs):
    for image in train_loader:
        output_img = model(image)
        loss = loss_func(output_img, image)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        inputs.append(image)
        outputs.append(output_img)
    
plt.style.use('fivethirtyeight')
plt.xlabel('Iterations')
plt.ylabel('Loss')

plt.plot(losses)
plt.show()

n = 5  # Number of images to show

# Get a batch of images
image_batch = inputs[0]  # First batch of original images
output_batch = outputs[0]  # First batch of reconstructed images

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Undo normalization
# mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
# std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

# image_batch = image_batch * std + mean  # Unnormalize
# output_batch = output_batch * std + mean  # Unnormalize

# Detach and clamp values for visualization
image_batch = torch.clamp(image_batch.detach(), 0, 1)
output_batch = torch.clamp(output_batch.detach(), 0, 1)

# Plot images side by side
fig, axes = plt.subplots(2, n, figsize=(n * 2, 4))

for i in range(n):
    # Original Image
    axes[0, i].imshow(image_batch[i].permute(1, 2, 0).cpu().numpy())  # Convert to NumPy
    axes[0, i].axis("off")

    # Reconstructed Image
    axes[1, i].imshow(output_batch[i].permute(1, 2, 0).cpu().numpy())  # Convert to NumPy
    axes[1, i].axis("off")

axes[0, 0].set_title("Original")
axes[1, 0].set_title("Reconstructed")

plt.show()


        



    

        

        

        


