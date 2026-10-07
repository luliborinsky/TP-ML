import torch
import torchvision.transforms as transforms
import torchvision
from net import Net

def main():
    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Prepare test data
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

    test_data = torchvision.datasets.ImageFolder('./test_images', transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False, num_workers=1)

    # Instantiate and load CNN
    net = Net()

    try:
        # Load pretrained weights
        net.load_state_dict(torch.load('face_net.pt', map_location=device, weights_only=True))
        print("Modelo 'face_net.pt' cargado exitosamente.")
    except Exception as e:
        print(f"Modelo no encontrado: {e}")

    net.to(device)
    net.eval()  

    print(f"Probando la CNN en: {device}...")

    correct = 0
    total = 0

    # Evaluate
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('-' * 40)
    print('Accuracy (Precisión) de la CNN en el test set: %.2f %%' % (100 * correct / total))
    print('-' * 40)

if __name__ == '__main__':
    main()
