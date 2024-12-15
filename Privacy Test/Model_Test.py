import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np

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

def process_image(image_path, model, transform, device, class_names):
    """Process a single image and predict its class."""
    try:
        image = Image.open(image_path).convert('L')  
        image = transform(image).unsqueeze(0).to(device) 
        
        with torch.no_grad():
            output = model(image)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
            
        print(f"Image: {os.path.basename(image_path)}")
        print(f"Predicted class: {predicted_class} ({class_names[predicted_class]})")
        print(f"Confidence: {confidence * 100:.2f}%\n")
        
        return predicted_class, confidence

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None, None

def main():
    if not torch.cuda.is_available():
        raise Exception("GPU not detected. Please ensure CUDA is installed and a GPU is available.")
    device = torch.device("cuda")

    transform = transforms.Compose([
        transforms.Grayscale(),          
        transforms.Resize((28, 28)),    
        transforms.ToTensor(),           
        transforms.Normalize((0.5,), (0.5,))  
    ])

    model = CNN_LSTM_Classifier(num_classes=2).to(device)
    model_path = 'cnn_lstm_model_old.pth' # Change model here
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} does not exist. Please ensure the model is trained and saved.")
    model.load_state_dict(torch.load(model_path))
    model.eval()  
    print(f"Model successfully loaded from {model_path}\n")

    # Class labels
    class_names = ['NONVPN', 'VPN']
    class_names = ['VPN','NONVPN'] # Old models
    label_to_index = {'nonvpn': 0, 'vpn': 1}

    source_dirs = ['test/nonvpn', 'test/tls'] 

    total_images = 0
    correct_predictions = 0
    vpn_count = 0
    nonvpn_count = 0
    true_labels = []
    predicted_labels = []

    count = 0
    for source_dir in source_dirs:
        if not os.path.isdir(source_dir):
            raise FileNotFoundError(f"Directory {source_dir} does not exist. Please provide a valid image folder path.")

        true_label_name = ['nonvpn','vpn']
        true_label = label_to_index.get(true_label_name[count])
        if true_label is None:
            raise ValueError(f"Cannot determine true label from directory name '{true_label_name}'. Expected 'vpn' or 'nonvpn'.")

        supported_formats = ('.png', '.jpg', '.jpeg', '.bmp')  
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                if file.lower().endswith(supported_formats):
                    image_path = os.path.join(root, file)
                    predicted_class, confidence = process_image(image_path, model, transform, device, class_names)
                    if predicted_class is not None:
                        total_images += 1
                        true_labels.append(true_label)
                        predicted_labels.append(predicted_class)
                        if predicted_class == true_label:
                            correct_predictions += 1
                        if predicted_class == 0:
                            nonvpn_count += 1
                        elif predicted_class == 1:
                            vpn_count += 1
        count = 1

    if total_images > 0:
        accuracy = correct_predictions / total_images * 100
        print(f"Total images processed: {total_images}")
        print(f"Number of correct predictions: {correct_predictions}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Number of NONVPN predictions: {nonvpn_count}")
        print(f"Number of VPN predictions: {vpn_count}")
    else:
        print("No images were processed.")
        return


    confusion_matrix = [[0, 0], [0, 0]]  
    for t, p in zip(true_labels, predicted_labels):
        confusion_matrix[t][p] += 1

    class_names = ['NONVPN', 'VPN']
    cm = np.array(confusion_matrix)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    # Annotate cells with counts
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()  

if __name__ == '__main__':
    main()
