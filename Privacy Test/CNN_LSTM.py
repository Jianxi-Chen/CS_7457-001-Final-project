import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

transform = transforms.Compose([
    transforms.Grayscale(),          
    transforms.Resize((28, 28)),     
    transforms.ToTensor(),           
    transforms.Normalize((0.5,), (0.5,))  
])

train_dir = 'dataset/Train'
test_dir = 'dataset/Test'

train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
test_dataset  = datasets.ImageFolder(root=test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

#Model structure
class CNN_LSTM_Classifier(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN_LSTM_Classifier, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.flatten = nn.Flatten(start_dim=2)
        self.lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=2, batch_first=True, dropout=0.5)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.cnn(x)        
        x = self.flatten(x)    
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out[:, -1, :])
        out = self.fc(lstm_out)
        return out

if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise Exception("Nndetect GPU")
    device = torch.device("cuda")
    print(f"Using device: {device}")

    model = CNN_LSTM_Classifier(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 200
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

    model_path = 'cnn_lstm_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')
    
    # optimizer_path = 'optimizer.pth'
    # torch.save(optimizer.state_dict(), optimizer_path)
    # print(f'Optimizer state saved to {optimizer_path}')
