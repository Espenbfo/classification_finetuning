from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import *
from torch.utils.data import random_split
def load_datasets(folder: Path, target_size=(224,224), train_fraction=0.8):
    if (train_fraction == 1):
        return ImageFolder(folder, transform=Compose([RandomHorizontalFlip(), RandomResizedCrop(target_size, (0.8, 1.0)), ToTensor()])), None
    return random_split(ImageFolder(folder, transform=Compose([RandomHorizontalFlip(), RandomResizedCrop(target_size, (0.8, 1.0)), ToTensor()])), (train_fraction, 1-train_fraction))

def load_dataloader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset, batch_size, shuffle)