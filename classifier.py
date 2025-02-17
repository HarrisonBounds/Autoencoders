from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torchvision import transforms
import torch
from aux_model import AE
import matplotlib.pyplot as plt
import random

# Lists to store the dataset, and extract the face and fist splits
dataset_split = []
face_split = []
fist_split = []

dataset = load_dataset("valhalla/emoji-dataset")

# Split data by text into face and fist samples
for data in dataset["train"]:
    text_label = data["text"]
    if text_label.find("face") != -1:
        face_split.append(data)
    if text_label.find("fist") != -1:
        fist_split.append(data)

# Transformations to augment the dataset
transformations = transforms.Compose([
    transforms.Resize(64),
    transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(30),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.ToTensor(),
])

# Increase fist samples, to make them equivalent to face samples
def augment_images(dataset, transform, target_size):
    augmented_data = []
    while len(augmented_data) < target_size:
        data = random.choice(dataset)
        image = data["image"].convert("RGB")
        augmented_data.append(transform(image))
    return augmented_data

face_split_transformed = augment_images(face_split, transformations, target_size=len(face_split) * 2)
fist_split_transformed = augment_images(fist_split, transformations, target_size=len(face_split) * 2)

# Store combined dataset into single list
dataset_split = face_split_transformed + fist_split_transformed

# Create labels based on length of transformed splits
labels = [0] * len(face_split_transformed) + [1] * len(fist_split_transformed)

# Split dataset into train, validation, and test sets
train_set, temp_set, train_labels, temp_labels = train_test_split(dataset_split, labels, test_size=0.4)
test_set, valid_set, test_labels, valid_labels = train_test_split(temp_set, temp_labels, test_size=0.5)

print(f"Train Set: ", len(train_set))
print(f"Validation Set: ", len(valid_set))
print(f"Test Set: ", len(test_set))

# Function to display images
def show_images(images, labels, title, num_samples=5):
    plt.figure(figsize=(15, 3))
    plt.suptitle(title, fontsize=16)
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(images[i].permute(1, 2, 0))  # Convert tensor to HWC format
        plt.title(f"Label: {labels[i]}")
        plt.axis("off")
    plt.show()

# Convert lists to tensors
train_set = torch.stack(train_set)
valid_set = torch.stack(valid_set)
test_set = torch.stack(test_set)

# Store image labels
train_labels = torch.tensor(train_labels, dtype=torch.long)
valid_labels = torch.tensor(valid_labels, dtype=torch.long)
test_labels = torch.tensor(test_labels, dtype=torch.long)

# show_images(train_set, train_labels, "Training Set Samples")

train_loader = torch.utils.data.DataLoader(dataset=list(zip(train_set, train_labels)), batch_size=32, shuffle=True)

cls_lambda = 0.2
num_classes = 2
input_height = 64
input_width = 64
num_channels = 3

# Initialising the model parameters
model = AE(input_width, input_height, num_channels, num_classes, cls_lambda)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay=1e-5)

num_epochs = 50

valid_loader = torch.utils.data.DataLoader(dataset=list(zip(valid_set, valid_labels)), batch_size=32, shuffle=False)

def compute_loss(decoded_imgs, class_logits, target_imgs, target_labels, cls_lambda):
    ae_loss = torch.nn.functional.mse_loss(decoded_imgs, target_imgs)
    cls_loss = torch.nn.functional.cross_entropy(class_logits, target_labels)
    total_loss = ae_loss + cls_lambda * cls_loss
    return total_loss, ae_loss, cls_loss
       
train_ae_losses = []
train_cls_losses = []
val_ae_losses = []
val_cls_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    # Training Loop
    model.train()
    epoch_ae_loss = 0.0
    epoch_cls_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        decoded_imgs, class_logits = model(images)
        total_loss, ae_loss, cls_loss = compute_loss(decoded_imgs, class_logits, images, labels, cls_lambda)

        total_loss.backward()
        optimizer.step()

        epoch_ae_loss += ae_loss.item()
        epoch_cls_loss += cls_loss.item()

    train_ae_losses.append(epoch_ae_loss / len(train_loader))
    train_cls_losses.append(epoch_cls_loss /len (train_loader))

    # Validation Loop
    model.eval()
    t_loss, t_ae_loss, t_cls_loss = 0.0, 0.0, 0.0
    correct, total = 0,0

    with torch.no_grad():
        for images, labels in valid_loader:
            
            decoded_imgs, class_logits = model(images)
            loss, ae_loss, cls_loss = compute_loss(decoded_imgs, class_logits, images, labels, cls_lambda)

            t_loss += loss.item()
            t_ae_loss += ae_loss.item()
            t_cls_loss += cls_loss.item()

            _, predicted = torch.max(class_logits.detach(), 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_ae_losses.append(t_ae_loss / len(valid_loader))
    val_cls_losses.append(t_cls_loss / len(valid_loader)) 
    accuracy = correct / total if total > 0 else 0
    val_accuracies.append(accuracy)


    print(f"Epoch {epoch+1}/{num_epochs}, "
          f"Train AE Loss: {(epoch_ae_loss / len(train_loader)):.4f}, "
          f"Train CLS Loss: {(epoch_cls_loss / len(train_loader)):.4f}, "
          f"Val AE Loss: {(t_ae_loss / len(valid_loader)):.4f}, "
          f"Val CLS Loss: {(t_cls_loss / len(valid_loader)):.4f}, "
          f"Val Accuracy: {accuracy:.4f}")
    

plt.figure(figsize=(18, 6))

# MSE Loss Plot
plt.subplot(1, 3, 1)
plt.plot(train_ae_losses, label="Train MSE Loss", color='blue')
plt.plot(val_ae_losses, label="Validation MSE Loss", color='red', linestyle="dashed")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title("Autoencoder Learning Curve")
plt.legend()

# Classification Loss Plot
plt.subplot(1, 3, 2)
plt.plot(train_cls_losses, label="Train Classification Loss", color='blue')
plt.plot(val_cls_losses, label="Validation Classification Loss", color='red', linestyle="dashed")
plt.xlabel("Epochs")
plt.ylabel("Cross-Entropy Loss")
plt.title("Classification Learning Curve")
plt.legend()

# Accuracy Plot
plt.subplot(1, 3, 3)
plt.plot(val_accuracies, label="Validation Accuracy", color='green')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig("loss_plots/classifier/loss_vs_epochs.png")
plt.show()

model.eval()
with torch.no_grad():
    images, labels = next(iter(valid_loader))
    reconstructed, class_logits = model(images)

    # Convert predictions to class labels
    predicted_labels = torch.argmax(class_logits, dim=1)

# Display first 5 images
n = 5
fig, axes = plt.subplots(3, n, figsize=(n * 2, 6))

for i in range(n):
    # Original Image
    axes[0, i].imshow(images[i].cpu().permute(1, 2, 0).numpy())
    axes[0, i].axis("off")

    # Reconstructed Image
    axes[1, i].imshow(reconstructed[i].cpu().permute(1, 2, 0).numpy())
    axes[1, i].axis("off")

    # True vs Predicted Labels
    axes[2, i].text(0.5, 0.5, f"True: {labels[i].item()}\nPred: {predicted_labels[i].item()}",
                    fontsize=12, ha='center', va='center')
    axes[2, i].axis("off")

axes[0, 0].set_title("Original")
axes[1, 0].set_title("Reconstructed")
axes[2, 0].set_title("Labels")

plt.tight_layout()
plt.savefig("output_plots/classifier/classifier_test_input_vs_output.png")
plt.show()
