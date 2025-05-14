from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import base64
import re
import io

# === Flask setup ===
app = Flask(__name__)

# === Define CNN (must match training code exactly!) ===
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# === Load the trained model ===
model = CNN()
model.load_state_dict(torch.load('cnn_model.pt', map_location=torch.device('cpu')))
model.eval()

# === Web route ===
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        img_data = data['image']
        img_str = re.search(r'base64,(.*)', img_data).group(1)
        img_bytes = base64.b64decode(img_str)

        # Load image and preprocess
        img = Image.open(io.BytesIO(img_bytes)).convert('L')
        img = img.resize((28, 28), resample=Image.NEAREST)
        img_arr = np.array(img)
        img_arr = 255 - img_arr  # White background → black
        img_arr = img_arr / 255.0

        # Convert to PyTorch tensor
        tensor = torch.tensor(img_arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # shape: [1, 1, 28, 28]

        # Predict
        with torch.no_grad():
            output = model(tensor)
            prediction = int(torch.argmax(output, dim=1).item())

        return jsonify({'prediction': prediction})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
