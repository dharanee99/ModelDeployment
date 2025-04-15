import io
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File, APIRouter
from PIL import Image
import torchvision.transforms as transforms
from fastapi.middleware.cors import CORSMiddleware

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 4) 

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

cnn = CNN()
cnn.load_state_dict(torch.load("octmnist.pth", map_location=torch.device('cpu')))
cnn.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Class names for OCTMNIST
class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("L")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = cnn(image)
        predicted_class = torch.argmax(output, dim=1).item()
        class_probabilities = torch.softmax(output, dim=1).squeeze().numpy()

    predictions = {
        class_names[i]: f"{class_probabilities[i] * 100:.2f}%"
        for i in range(len(class_names))
    }

    sorted_predictions = {
        k: v for k, v in sorted(predictions.items(), key=lambda item: float(item[1][:-1]), reverse=True)
    }

    return {
        "predicted_class": class_names[predicted_class],
        "predictions": sorted_predictions
    }
