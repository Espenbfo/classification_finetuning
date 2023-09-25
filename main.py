import torch
from tqdm import tqdm

from dataset import load_dataset, load_dataloader
from model import init_model, load_model
from torch.optim import Adam
from torch.nn.functional import cross_entropy
import json
import time
import numpy as np
DEVICE = "cuda"
EPOCHS = 5
CONTINUE_TRAINING = True
LOSS_MEMORY = 300 # batches
BATCH_SIZE = 4
CHECKPOINT_TIME = 4 # Minutes
LEARNING_RATE_CLASSIFIER = 1e-3
LEARNING_RATE_FEATURES = 1e-4
FILENAME = "weights.pt"
TRAIN_TRANSFORMER = False
DATASET_FOLDER = Path(r"")


def main():
    print("Cuda available?", torch.cuda.is_available())
    dataset = load_dataset(DATASET_FOLDER)
    dataloader = load_dataloader(dataset, BATCH_SIZE, True)
    if CONTINUE_TRAINING:
        model = load_model(len(dataset.classes), "weights.pt").to(DEVICE)
    else:
        model = init_model(len(dataset.classes)).to(DEVICE)
    if TRAIN_TRANSFORMER:
        model.transformer.eval()
        optimizer_features = Adam(model.transformer.parameters(LEARNING_RATE_FEATURES))
    optimizer_classifier = Adam(model.classifier.parameters(LEARNING_RATE_CLASSIFIER))

    with open("classes.json", "w") as f:
        json.dump(dataset.classes, f)

    loss_arr = np.zeros(LOSS_MEMORY)
    acc_arr = np.zeros(LOSS_MEMORY)
    checkpoint_time = time.time()
    for epoch in range(EPOCHS):
        print("epoch", epoch+1)
        total_loss = 0
        for index, (batch,label) in (pbar := tqdm(enumerate(dataloader), total=len(dataloader))):
            batch = batch.to(DEVICE)
            label = label.to(DEVICE)
            result = model(batch)
            loss = cross_entropy(result, label)
            loss.backward()

            if TRAIN_TRANSFORMER:
                optimizer_features.step()
                optimizer_features.zero_grad()

            optimizer_classifier.step()
            optimizer_classifier.zero_grad()
            loss_arr = np.roll(loss_arr, -1)
            loss_arr[-1] = loss.detach().cpu()

            accuracy = torch.eq(label, torch.argmax(result, dim=1)).sum()/BATCH_SIZE

            acc_arr = np.roll(acc_arr, -1)
            acc_arr[-1] = accuracy

            pbar.postfix = f"mean loss the last {LOSS_MEMORY} batches {loss_arr.mean():.3f} | accuracy {acc_arr.mean():.3f} | time_since_checkpoint {time.time()-checkpoint_time:.1f}s"
            if (time.time() > checkpoint_time+CHECKPOINT_TIME*60):
                torch.save(model.state_dict(), FILENAME)
                checkpoint_time = time.time()



if __name__ == "__main__":
    main()