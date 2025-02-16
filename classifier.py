from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torchvision import transforms
import torch
from aux_model import AE
import matplotlib.pyplot as plt

dataset_split = []
man_split = []
woman_split = []

dataset = load_dataset("valhalla/emoji-dataset")

# Split data by text into man and woman samples
for data in dataset["train"]:
    text_label = data["text"]
    if text_label.find("man") != -1:
        man_split.append(data)
    if text_label.find("woman") != -1:
        woman_split.append(data)


transformations = transforms.Compose([
    transforms.Resize(64),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

# Augment woman data to increase samples
def augment_images(dataset, transform, augment_times):
    augmented_data = []
    for data in dataset:
        image = data["image"].convert("RGB")
        for _ in range(augment_times):
            augmented_data.append(transform(image))
    return augmented_data

man_split_transformed = augment_images(man_split, transformations, augment_times=1)
woman_split_transformed = augment_images(woman_split, transformations, augment_times=2)

dataset_split = man_split_transformed + woman_split_transformed

labels = [0] * len(man_split_transformed) + [1] * len(woman_split_transformed)


train_set, temp_set, train_labels, temp_labels = train_test_split(dataset_split, labels, test_size=0.4)
test_set, valid_set, test_labels, valid_labels = train_test_split(temp_set, temp_labels, test_size=0.5)

# Convert lists to tensors
train_set = torch.stack(train_set)
valid_set = torch.stack(temp_set)
test_set = torch.stack(test_set)

train_labels = torch.tensor(train_labels)
valid_labels = torch.tensor(valid_labels)
test_labels = torch.tensor(test_labels)

train_loader = torch.utils.data.DataLoader(dataset=list(zip(train_set, train_labels)), batch_size=32, shuffle=True)

# Initialising the model parameters
model = AE(64, 64, 3, 2, 0.9)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay=1e-6)

num_epochs = 50
total_losses = []
ae_losses = []
classification_losses = []

inputs = []
outputs = []

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        decoded_imgs, class_logits = model.forward(images)

        total_loss, ae_loss, classification_loss = model.compute_loss(images, images, labels)

        total_losses.append(total_loss.item())
        ae_losses.append(ae_loss.item())
        classification_losses.append(classification_loss.item())

        total_loss.backward()
        optimizer.step()

        # Store the first batch of the last epoch for visualization
        if epoch == num_epochs - 1 and i == 0:
            inputs.append(images)
            outputs.append(decoded_imgs)




plt.style.use('fivethirtyeight')
plt.xlabel('Iterations')
plt.ylabel('Loss')

plt.plot(classification_losses)
plt.savefig("loss_plots/training/loss_vs_iterations")
plt.show()

n = 5

# Get a batch of images
image_batch = inputs[0]  # First batch of original images
output_batch = outputs[0]  # First batch of reconstructed images

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Detach and clamp values for visualization
image_batch = torch.clamp(image_batch.detach(), 0, 1)
output_batch = torch.clamp(output_batch.detach(), 0, 1)

# Plot images side by side
fig, axes = plt.subplots(2, n, figsize=(n * 2, 4))

for i in range(n):
    # Original Image
    axes[0, i].imshow(image_batch[i].permute(
        1, 2, 0).cpu().numpy())  # Convert to NumPy
    axes[0, i].axis("off")

    # Reconstructed Image
    axes[1, i].imshow(output_batch[i].permute(
        1, 2, 0).cpu().numpy())  # Convert to NumPy
    axes[1, i].axis("off")

axes[0, 0].set_title("Original")
axes[1, 0].set_title("Reconstructed")

plt.savefig("output_plots/training/input_vs_output.png")
plt.show()
