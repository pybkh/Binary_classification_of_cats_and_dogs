import torch  
import torch.nn as nn  
import torch.optim as optim  
from torchvision import datasets, transforms, models  
from torch.utils.data import DataLoader, random_split  
import matplotlib.pyplot as plt  
  
transform = transforms.Compose([  
    transforms.Resize((224, 224)),     # ResNet 输入 224x224    transforms.ToTensor(),  
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  
                         [0.229, 0.224, 0.225])   # ResNet 官方均值方差  
])  
  
dataset = datasets.ImageFolder(root='./cat_dog', transform=transform)  
  
train_size = int(0.8 * len(dataset))  
val_size = int(0.1 * len(dataset))  
test_size = len(dataset) - train_size - val_size  
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])  
  
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)  
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  
  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
  
model = models.resnet18(pretrained=True)  
  
# 是否冻结预训练参数  
for param in model.parameters():  
    param.requires_grad = False  
  
# 替换最后一层  
num_features = model.fc.in_features  
model.fc = nn.Sequential(  
    nn.Linear(num_features, 1),  
    nn.Sigmoid()  
)  
  
model = model.to(device)  
  
criterion = nn.BCELoss()  
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)  
  
epochs = 10  
train_losses = []  
train_accs = []  
val_accs = []  
  
for epoch in range(epochs):  
    model.train()  
    epoch_train_loss = 0  
    correct_train = 0  
    total_train = 0  
  
    for images, labels in train_loader:  
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)  
  
        outputs = model(images)  
        loss = criterion(outputs, labels)  
  
        optimizer.zero_grad()  
        loss.backward()  
        optimizer.step()  
  
        epoch_train_loss += loss.item()  
        preds = (outputs > 0.5).int()  
        correct_train += (preds == labels.int()).sum().item()  
        total_train += labels.size(0)  
  
    avg_train_loss = epoch_train_loss / len(train_loader)  
    train_acc = correct_train / total_train  
  
    train_losses.append(avg_train_loss)  
    train_accs.append(train_acc)  
  
    model.eval()  
    correct_val = 0  
    total_val = 0  
  
    with torch.no_grad():  
        for images, labels in val_loader:  
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)  
            outputs = model(images)  
            preds = (outputs > 0.5).int()  
            correct_val += (preds == labels.int()).sum().item()  
            total_val += labels.size(0)  
  
    val_acc = correct_val / total_val  
    val_accs.append(val_acc)  
  
    print(f"轮次 [{epoch+1}/{epochs}]  "  
          f"训练损失: {avg_train_loss:.4f}  "  
          f"训练准确率: {train_acc:.4f}  "  
          f"验证准确率: {val_acc:.4f}")  

torch.save(model.state_dict(), 'resnet18_cat_dog.pth')
print("模型参数已成功保存为 resnet18_cat_dog.pth")
  
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False  
plt.figure(figsize=(10,5))  
plt.plot(train_losses, label='训练损失')  
plt.plot(train_accs, label='训练准确率')  
plt.plot(val_accs, label='验证准确率')  
plt.legend()  
plt.grid(True)  
plt.show()  
  
  
model.eval()  
correct = 0  
total = 0  
  
with torch.no_grad():  
    for images, labels in test_loader:  
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)  
        outputs = model(images)  
        preds = (outputs > 0.5).int()  
        correct += (preds == labels.int()).sum().item()  
        total += labels.size(0)  
  
print(f"测试准确率: {correct / total:.4f}")
