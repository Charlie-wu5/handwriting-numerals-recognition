import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ✅ 定义 CNN 模型
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)  # 输入通道1，输出通道16，卷积核3x3
        self.pool = nn.MaxPool2d(2, 2)               # 池化层 2x2
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1) # 第二层卷积
        self.fc1 = nn.Linear(32 * 7 * 7, 128)        # 全连接层，7x7 是经过两次池化后的大小
        self.fc2 = nn.Linear(128, 10)                # 输出10类

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))         # [1,28,28] → [16,14,14]
        x = self.pool(F.relu(self.conv2(x)))         # [16,14,14] → [32,7,7]
        x = x.view(-1, 32 * 7 * 7)                    # 展平为向量
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ✅ 数据加载
transform = transforms.Compose([
    transforms.ToTensor(),  # 转为 0~1 范围的张量
    transforms.Normalize((0.1307,), (0.3081,))  # 标准化
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# ✅ 模型初始化
model = CNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ✅ 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ✅ 训练循环
for epoch in range(5):  # 训练5轮
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}")

# ✅ 验证准确率
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {correct / total:.4f}")

# ✅ 保存模型
torch.save(model.state_dict(), 'cnn_model.pt')
print("模型已保存为 cnn_model.pt")
