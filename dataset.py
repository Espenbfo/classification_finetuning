from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import *
def load_dataset(folder: Path | str, target_size=(224,224)):
    return ImageFolder(folder, transform=Compose([RandomHorizontalFlip(), RandomResizedCrop(target_size, (0.8, 1.0)), ToTensor()]))

def load_dataloader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset, batch_size, shuffle)