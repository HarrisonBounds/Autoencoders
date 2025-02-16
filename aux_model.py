import torch.nn as nn


class AE(nn.Module):
    def __init__(self, input_width, input_height, num_channels, num_classes, classification_lambda = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.classification_lambda = classification_lambda

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 128, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, num_channels, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * (input_width // 32) * (input_height // 32), 128), # Stride of 2 reduces the image dimensions by 2, so for 5 layers image size is reduced by 32
            nn.ReLU(),
            nn.Linear(128, self.num_classes), # 2 stems from number of output classes
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        class_logits = self.classifier(encoded)
        return decoded, class_logits
    
    def compute_loss(self, x, target_images, target_labels):
        decoded_imgs, class_logits = self.forward(x)

        mse_loss = nn.MSELoss()
        ae_loss = mse_loss(decoded_imgs, target_images)

        ce_loss = nn.CrossEntropyLoss()
        classification_loss = ce_loss(class_logits, target_labels) 

        total_loss = ae_loss + self.classification_lambda * classification_loss
        
        return total_loss, ae_loss, classification_loss

