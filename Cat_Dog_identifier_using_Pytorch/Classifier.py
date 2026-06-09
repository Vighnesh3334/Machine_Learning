import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

# --- Keep definitions out in the open ---

path_train = r"C:\Users\Sudeep\Desktop\ML_data\dataset\training_set"
path_test = r"C:\Users\Sudeep\Desktop\ML_data\dataset\test_set"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=0, shear=20, scale=(0.8, 1.2)),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1) 
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


# 1. Load Datasets and Create Data Loaders
training_set = datasets.ImageFolder(root=path_train, transform=train_transform)
test_set = datasets.ImageFolder(root=path_test, transform=test_transform)

# Changed num_workers to 0 to prevent multiprocessing crashes without the main block
train_loader = DataLoader(training_set, batch_size=32, shuffle=True, num_workers=0)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=0)

class_indices = training_set.class_to_idx
idx_to_class = {v: k for k, v in class_indices.items()}


cnn = CNN().to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(cnn.parameters(), lr=0.001)

epochs = 25

# 3. Training and Validation Loops
for epoch in range(epochs):
    cnn.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
        
        
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        predictions = (outputs > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
       
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    cnn.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            outputs = cnn(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * images.size(0)
            predictions = (outputs > 0.5).float()
            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)
            
    print(f"Epoch {epoch+1}/{epochs} -> "
          f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | "
          f"Val Loss: {(val_loss/val_total):.4f} | Val Acc: {(val_correct/val_total):.4f}")


# 4. Single Prediction Evaluation
def predict_single_image(image_path):
    cnn.eval()
    try:
        img = Image.open(image_path)
        img_transformed = test_transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            result = cnn(img_transformed).item()
            
        pred_idx = 1 if result > 0.5 else 0
        prediction = idx_to_class[pred_idx]
        print(f"\nPrediction for {image_path}: {prediction} (Confidence Score: {result:.4f})")
    except FileNotFoundError:
        print(f"\nCould not find single prediction file at: {image_path}. Please check your path.")

# Run the single image prediction check after training completes
predict_single_image('dataset/single_prediction/cat_or_dog_1.jpg')