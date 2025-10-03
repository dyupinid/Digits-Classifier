import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

#Определение преобразований: нормализация и преобразование в тензоры
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  #Нормализация для одноканальных изображений
])

#Загрузка тренировочного и тестового наборов данных
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

#Создание DataLoader для батчевой обработки
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#Просмотр примеров изображений
def imshow(img):
    img = img / 2 + 0.5  #Денормализация
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
    plt.show()

#Получение одного батча
dataiter = iter(train_loader)
images, labels = next(dataiter)

#Вывод изображений
imshow(torchvision.utils.make_grid(images[:6]))
print('Labels:', labels[:6].tolist())

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),  #Вход: 28x28 пикселей
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)  #Выход: 10 классов (цифры 0-9)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNet()
print(model)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)

        #Forward pass
        outputs = model(data)
        loss = criterion(outputs, targets)

        #Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}')

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    print(f'Accuracy: {100 * correct / total:.2f}%')

    torch.save(model.state_dict(), 'mnist_model.pth')

    #Загрузка модели
    loaded_model = NeuralNet().to(device)
    loaded_model.load_state_dict(torch.load('mnist_model.pth'))
    loaded_model.eval()

    #Предсказание для одного изображения
    sample_image, sample_label = test_dataset[0]
    with torch.no_grad():
        output = loaded_model(sample_image.unsqueeze(0).to(device))
        predicted = torch.argmax(output).item()

    print(f'Predicted: {predicted}, Actual: {sample_label}')