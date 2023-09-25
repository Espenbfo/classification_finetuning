import torch
from dataset import load_datasets, load_dataloader
import cv2
from torchvision.transforms import ToPILImage
import numpy as np
import json
from model import load_model
def get_class(classes, pred):
    return classes[torch.argmax(pred)]

def main():
    with open("classes.json", "r") as f:
        classes = json.load(f)

    dataset, _ = load_datasets(r"", train_fraction=1.0)
    dataloader = load_dataloader(dataset, 1, True)
    model = load_model(len(classes), "weights.pt").to("cuda")
    model.eval()
    for batch, label in dataloader:
        result = model(batch.to("cuda")).to("cpu")
        print(result, label)
        image = np.array(ToPILImage()(batch[0]))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(get_class(classes, result))
        cv2.imshow("1", image)
        cv2.waitKey(0)

if __name__ == "__main__":
    main()