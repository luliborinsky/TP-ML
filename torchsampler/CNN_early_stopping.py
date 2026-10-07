import os
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data.sampler import WeightedRandomSampler

# change
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(BASE_DIR, 'train_images')
test_dir = os.path.join(BASE_DIR, 'test_images')


def load_all(folder, cache_file):
    if os.path.exists(cache_file):
        d = torch.load(cache_file)
        return d['x'], d['y']
    ds = torchvision.datasets.ImageFolder(folder)
    arr = np.stack([np.array(Image.open(path).convert('L')) for path, _ in ds.samples])
    x = torch.from_numpy(arr).unsqueeze(1)
    y = torch.tensor(ds.targets)
    torch.save({'x': x, 'y': y}, cache_file)
    return x, y

# change
train_x, train_y = load_all(train_dir, os.path.join(BASE_DIR, 'cache_train.pt'))
test_x, test_y = load_all(test_dir, os.path.join(BASE_DIR, 'cache_test.pt'))


def norm(x):
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    s = x.std(dim=(1, 2, 3), keepdim=True)
    return (x - m) / (s + 1e-6)


def augment(x):

    B, _, H, W = x.shape
    flip = (torch.rand(B) < 0.5).view(B, 1, 1, 1)
    x = torch.where(flip, x.flip(3), x)

    # affine transformation (black background)
    ang = (torch.rand(B) * 2 - 1) * math.radians(10)
    sc = 0.9 + 0.2 * torch.rand(B)
    tx = (torch.rand(B) * 2 - 1) * 0.2
    ty = (torch.rand(B) * 2 - 1) * 0.2
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos, -sin, tx], dim=1),
                         torch.stack([sin, cos, ty], dim=1)], dim=1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

    # brightness and contrast
    b = 0.4 + 1.2 * torch.rand(B, 1, 1, 1)
    x = (x * b).clamp(0, 1)
    c = 0.4 + 1.2 * torch.rand(B, 1, 1, 1)
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    x = ((x - m) * c + m).clamp(0, 1)

    # blur
    sigma = torch.where(torch.rand(B) < 0.5, 0.1 + 1.9 * torch.rand(B), torch.full((B,), 0.01))
    e = torch.exp(-1.0 / (2 * sigma ** 2))
    k1 = torch.stack([e, torch.ones(B), e], dim=1)
    k1 = k1 / k1.sum(dim=1, keepdim=True)
    k2 = k1.unsqueeze(2) * k1.unsqueeze(1)
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

# validation and test: normalized ONCE and left ready (no augmentation)
valid_idx_t = torch.tensor(valid_idx)
X_val = norm(train_x[valid_idx_t].float() / 255)
y_val = train_y[valid_idx_t]
X_test = norm(test_x.float() / 255)

counts = torch.bincount(train_y[train_new_idx])
weights = torch.zeros(len(train_y), dtype=torch.double)
weights[train_new_idx] = 1.0 / counts[train_y[train_new_idx]].double()
trainsampler = WeightedRandomSampler(weights, num_samples=len(train_new_idx), replacement=True)

trainloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_y),
                                          batch_size=batch_size, sampler=trainsampler, num_workers=0)
validloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_val, y_val),
                                          batch_size=1024, num_workers=0)
testloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test, test_y),
                                         batch_size=batch_size, shuffle=True, num_workers=0)
classes = ('noface','face')

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy().clip(0, 1)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Net(nn.Module):# We define the neural network class
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5) # Grayscale (a single channel), 6 outputs, 5x5 analysis area
        self.pool = nn.MaxPool2d(2, 2)# Size reduction of the information (2x2 pooling) keeping the important information
        self.conv2 = nn.Conv2d(6, 16, 5)# 6 inputs, 16 outputs, 5x5 analysis area
        self.fc1 = nn.Linear(16 * 6 * 6, 120)# We reduce from 16 outputs *6*6 of analysis area to 120 neurons
        self.fc2 = nn.Linear(120, 84)# We reduce to 84 neurons
        self.fc3 = nn.Linear(84, 2)# We reduce to 2 neurons

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # We pool the first convolution
        x = self.pool(F.relu(self.conv2(x))) # We pool the second convolution

        # Here we already pull out the important patterns

        x = torch.flatten(x, 1) # flatten all dimensions except batch, because we have to
        x = F.relu(self.fc1(x)) # Final predictions of the first layer
        x = F.relu(self.fc2(x)) # Final predictions of the second layer
        x = self.fc3(x)# Overall final predictions
        return x # returns the prediction


net = Net()

criterion = nn.CrossEntropyLoss() # we create a loss detector (entropy)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# --- AUC and per-class recall
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


# that early stopping
max_epochs = 30          # max here
patience = 5             # waiting for improvment
best_auc = -1.0
best_state = None
best_epoch = 0
epochs_no_improve = 0

for epoch in range(max_epochs):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = prep_train(inputs)    # augmentation + normalization of the batch (vectorized)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs) # We feed the inputs to the network, labels and image
        loss = criterion(outputs, labels) # We get the entropy
        loss.backward() # We look at the entropy of the last analysis
        optimizer.step() # Change filters and update

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 100 micro-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

    # at the end of every epoch: AUC and per-class recall on validation and test
    v_auc, v_r0, v_r1 = evaluate(validloader)
    t_auc, t_r0, t_r1 = evaluate(testloader)   # just for info
    print(f'epoch {epoch + 1} | VAL auc {v_auc:.3f} rec0 {v_r0:.3f} rec1 {v_r1:.3f} '
          f'| test auc {t_auc:.3f} rec0 {t_r0:.3f} rec1 {t_r1:.3f}')

    # check f its improved
    if v_auc > best_auc:
        best_auc = v_auc
        best_epoch = epoch + 1
        best_state = copy.deepcopy(net.state_dict())
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f'early stopping in epoch {epoch + 1}')
            break

print(f'finished training. best {best_epoch}, VAL auc = {best_auc:.3f}')

# best weight before give back
net.load_state_dict(best_state)

PATH = os.path.join(BASE_DIR, 'cifar_net.pt')
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

print(f'Accuracy of the network on the test images: {100 * correct // total} %')

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
