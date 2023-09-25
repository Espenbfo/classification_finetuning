import shutil
from pathlib import Path
import json

import torch
from torch.nn.functional import softmax
from PIL import Image
from torchvision.transforms import *
import tqdm
from model import load_model

INPUT_FOLDER = Path(r"")
OUTPUT_FOLDER = Path("")

with open("classes.json", "r") as f:
    classes = json.load(f)

OUTPUT_FOLDER.mkdir(exist_ok=True)
for class_name in classes:
    (OUTPUT_FOLDER / class_name).mkdir(exist_ok=True)
model = load_model(len(classes), "weights.pt").to("cuda")
model.eval()
transforms = Compose([Resize((224,224)), ToTensor()])
for file in tqdm.tqdm(list(INPUT_FOLDER.rglob("*"))):
    image = Image.open(file).convert("RGB")
    batch = transforms(image).unsqueeze(0)
    result = model(batch.to("cuda")).to("cpu")
    scores = softmax(result)
    maxarg= torch.argmax(scores[0])
    pred_class = classes[maxarg]
    file_destination = OUTPUT_FOLDER / pred_class / f"{scores[0][maxarg]:.2f}_{file.stem}{file.suffix}"
    shutil.copy(file, file_destination)