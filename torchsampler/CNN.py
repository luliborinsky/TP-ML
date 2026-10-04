#IMBALANCED SAMPLER

import torch
import torch.utils.data
import torchvision


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        callback_get_label func: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices=None, num_samples=None, callback_get_label=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
                
        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        if self.callback_get_label:
            return self.callback_get_label(dataset, idx)
        elif isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels[idx].item()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return dataset.imgs[idx][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[idx][1]
        else:
            raise NotImplementedError
                
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples











import os
import math
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data.sampler import SubsetRandomSampler

train_dir = r'C:\Users\Samue\OneDrive\Desktop\INSA LYON\Semestre de Intercambio\ML\deep_learning_project\deep_learning_project\train_images'

test_dir = r'C:\Users\Samue\OneDrive\Desktop\INSA LYON\Semestre de Intercambio\ML\deep_learning_project\deep_learning_project\test_images'

import torch.nn as nn
import torch.nn.functional as F



def load_all(folder, cache_file):
    if os.path.exists(cache_file):
        d = torch.load(cache_file)
        return d['x'], d['y']
    ds = torchvision.datasets.ImageFolder(folder)
    print(folder, ds.class_to_idx)
    arr = np.stack([np.array(Image.open(path).convert('L')) for path, _ in ds.samples]) 
    x = torch.from_numpy(arr).unsqueeze(1)                                                
    y = torch.tensor(ds.targets)
    torch.save({'x': x, 'y': y}, cache_file)
    return x, y

train_x, train_y = load_all(train_dir, 'cache_train.pt')
test_x, test_y = load_all(test_dir, 'cache_test.pt')


def norm(x):
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    s = x.std(dim=(1, 2, 3), keepdim=True)
    return (x - m) / (s + 1e-6)



def augment(x):
    # x: (B,1,36,36) float en [0,1]
    B, _, H, W = x.shape

    # 1) flip horizontal con probabilidad 0.5
    flip = (torch.rand(B) < 0.5).view(B, 1, 1, 1)
    x = torch.where(flip, x.flip(3), x)

    # 2) transformación afín (fondo negro)
    ang = (torch.rand(B) * 2 - 1) * math.radians(10)
    sc = 0.9 + 0.2 * torch.rand(B)
    tx = (torch.rand(B) * 2 - 1) * 0.2     # 10 % del ancho = 0.2 en coordenadas normalizadas [-1, 1]
    ty = (torch.rand(B) * 2 - 1) * 0.2
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos, -sin, tx], dim=1),
                         torch.stack([sin, cos, ty], dim=1)], dim=1)       # (B,2,3)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

    # 3) brillo y contraste (con recorte a [0,1], como ColorJitter)
    b = 0.4 + 1.2 * torch.rand(B, 1, 1, 1)
    x = (x * b).clamp(0, 1)
    c = 0.4 + 1.2 * torch.rand(B, 1, 1, 1)
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    x = ((x - m) * c + m).clamp(0, 1)

    # 4) desenfoque gaussiano 3x3 en la mitad de las imágenes (sigma pequeño = sin desenfoque)
    sigma = torch.where(torch.rand(B) < 0.5, 0.1 + 1.9 * torch.rand(B), torch.full((B,), 0.01))
    e = torch.exp(-1.0 / (2 * sigma ** 2))                 # peso de los vecinos (el central vale 1)
    k1 = torch.stack([e, torch.ones(B), e], dim=1)         # (B,3)
    k1 = k1 / k1.sum(dim=1, keepdim=True)
    k2 = k1.unsqueeze(2) * k1.unsqueeze(1)                 # (B,3,3)
    xp = F.pad(x.reshape(1, B, H, W), (1, 1, 1, 1), mode='reflect')
    x = F.conv2d(xp, k2.unsqueeze(1), groups=B).reshape(B, 1, H, W)
    return x


def prep_train(x_uint8):
    return norm(augment(x_uint8.float() / 255))


valid_size = 0.2
batch_size = 32

num_train = len(train_y)
indices_train = list(range(num_train))
np.random.shuffle(indices_train)
split_tv = int(np.floor(valid_size * num_train))
train_new_idx, valid_idx = indices_train[split_tv:],indices_train[:split_tv]

from collections import Counter

print("Training distribution:")
print(Counter(train_y[train_new_idx].tolist()))

# validación y test: se normalizan UNA vez y se dejan listos (sin aumentos)
valid_idx_t = torch.tensor(valid_idx)
X_val = norm(train_x[valid_idx_t].float() / 255)
y_val = train_y[valid_idx_t]
X_test = norm(test_x.float() / 255)

train_labels = train_y.tolist()
trainsampler = ImbalancedDatasetSampler(train_labels, indices=train_new_idx,
                                        callback_get_label=lambda ds, i: ds[i])

trainloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_y),
                                          batch_size=batch_size, sampler=trainsampler, num_workers=0)
validloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_val, y_val),
                                          batch_size=1024, num_workers=0)
testloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test, test_y),
                                         batch_size=batch_size, shuffle=True, num_workers=0)
classes = ('noface','face')

# for epoch in range(1, n_epochs+1):
# for data, target in train_loader:

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)
images = prep_train(images)      # los lotes de entrenamiento llegan en bruto: aplicar aumentos + normalización

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))


class Net(nn.Module):#Definimos clase de la red neuronal
    def __init__(self):
        super().__init__() 
        self.conv1 = nn.Conv2d(1, 6, 5) # Grayscale (un solo canal), 6 outputs, 5x5 de área de análisis
        self.pool = nn.MaxPool2d(2, 2)#Reducción de tamaño de la información (pooling de 5x5 a 2x2) con la información importante
        self.conv2 = nn.Conv2d(6, 16, 5)#6 inputs, 16 outputs, 5x5 de área de análisis
        self.fc1 = nn.Linear(16 * 6 * 6, 120)#Reducimos de 16 outputs *5*5 de área de análisis a 120 neuronas
        self.fc2 = nn.Linear(120, 84)#Reducimos a 84 neuronas
        self.fc3 = nn.Linear(84, 2)#Reducimos a 10 neuronas

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # Hacemos pool a la primera convolución
        x = self.pool(F.relu(self.conv2(x))) # Hacemos pool a la segunda convolución
        
        #Aquí ya sacamos los patrones importantes
        
        x = torch.flatten(x, 1) # flatten all dimensions except batch, porque toca
        x = F.relu(self.fc1(x)) #Predicciones finales de la primera capa
        x = F.relu(self.fc2(x)) #Predicciones finales de la segunda capa
        x = self.fc3(x)#Predicciones finales totales
        return x # devuelve la predicción


net = Net()


import torch.optim as optim

criterion = nn.CrossEntropyLoss() # creamos un detector de loss (entropía)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# --- AUC y recall por clase (sin depender de sklearn) ---
def auc_score(y, p):
    order = np.argsort(p)
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    n1 = y.sum()
    n0 = len(y) - n1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def evaluate(loader):
    net.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            ps.append(F.softmax(net(x), dim=1)[:, 1])
            ys.append(y)
    net.train()
    y = torch.cat(ys).numpy()
    p = torch.cat(ps).numpy()
    pred = (p > 0.5).astype(int)
    rec0 = (pred[y == 0] == 0).mean()
    rec1 = (pred[y == 1] == 1).mean()
    return auc_score(y, p), rec0, rec1


for epoch in range(10):  # 10 épocas

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = prep_train(inputs)    # aumentos + normalización del lote (vectorizado)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs) #Le ponemos los imputs a la red, labels e imágen
        loss = criterion(outputs, labels) #Sacamos la entropía
        loss.backward() # Vemos la entropía del último análisis
        optimizer.step() #Nos devolvemos

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:    # print every 10 micro-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}')
            running_loss = 0.0

    # al final de cada época: AUC y recall por clase en validación y test
    v_auc, v_r0, v_r1 = evaluate(validloader)
    t_auc, t_r0, t_r1 = evaluate(testloader)
    print(f'=== época {epoch + 1} | VAL auc {v_auc:.3f} rec0 {v_r0:.3f} rec1 {v_r1:.3f} '
          f'| TEST auc {t_auc:.3f} rec0 {t_r0:.3f} rec1 {t_r1:.3f}')

print('Finished Training')

PATH = './cifar_net.pt'
torch.save(net.state_dict(), PATH)



dataiter = iter(testloader)
images, labels = next(dataiter)

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

net = Net()
net.load_state_dict(torch.load(PATH, weights_only=True))

outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(4)))

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

