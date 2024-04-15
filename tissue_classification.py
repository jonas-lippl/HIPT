import os
import random

import matplotlib.pyplot as plt
import pamly
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

"""
screen -dmS tissue_classification sh -c 'docker run --shm-size=400gb --gpus \"device=0\" --name jol_job1  -it --rm -u `id -u $USER` -v /sybig/projects/FedL/data:/data -v /sybig/home/jol/Code/HIPT:/mnt jol_hipt python3 /mnt/tissue_classification.py; exec bash'
screen -dmS binary_tissue_classification sh -c 'docker run --shm-size=400gb --gpus \"device=0\" --name jol_job1  -it --rm -u `id -u $USER` -v /sybig/projects/FedL/data:/data -v /sybig/home/jol/Code/HIPT:/mnt jol_hipt python3 /mnt/tissue_classification.py; exec bash'
screen -dmS binary_tissue_classification_BCE sh -c 'docker run --shm-size=400gb --gpus \"device=0\" --name jol_job1  -it --rm -u `id -u $USER` -v /sybig/projects/FedL/data:/data -v /sybig/home/jol/Code/HIPT:/mnt jol_hipt python3 /mnt/tissue_classification.py; exec bash'
screen -dmS binary_tissue_classification_BCE_smaller_model_slide_split sh -c 'docker run --shm-size=400gb --gpus \"device=0\" --name jol_job1  -it --rm -u `id -u $USER` -v /sybig/projects/FedL/data:/data -v /sybig/home/jol/Code/HIPT:/mnt jol_hipt python3 /mnt/tissue_classification.py; exec bash'
screen -dmS binary_tissue_classification_BCE_smaller_model_AdamW_0.01_weight_decay sh -c 'docker run --shm-size=400gb --gpus \"device=2\" --name jol_job3  -it --rm -u `id -u $USER` -v /sybig/projects/FedL/data:/data -v /sybig/home/jol/Code/HIPT:/mnt jol_hipt python3 /mnt/tissue_classification.py; exec bash'
screen -dmS binary_tissue_classification_BCE_dropout sh -c 'docker run --shm-size=400gb --gpus \"device=3\" --name jol_job2  -it --rm -u `id -u $USER` -v /sybig/projects/FedL/data:/data -v /sybig/home/jol/Code/HIPT:/mnt jol_hipt python3 /mnt/tissue_classification.py; exec bash'
screen -dmS binary_tissue_classification_BCE_0.5_dropout sh -c 'docker run --shm-size=400gb --gpus \"device=1\" --name jol_job2  -it --rm -u `id -u $USER` -v /sybig/projects/FedL/data:/data -v /sybig/home/jol/Code/HIPT:/mnt jol_hipt python3 /mnt/tissue_classification.py; exec bash'
screen -dmS binary_tissue_classification_BCE_0.5_dropout_patch_based_train_test_split sh -c 'docker run --shm-size=400gb --gpus \"device=2\" --name jol_job3  -it --rm -u `id -u $USER` -v /sybig/projects/FedL/data:/data -v /sybig/home/jol/Code/HIPT:/mnt jol_hipt python3 /mnt/tissue_classification.py; exec bash'
screen -dmS binary_tissue_classification_BCE_dropout_patch_based_train_test_split sh -c 'docker run --shm-size=400gb --gpus \"device=0\" --name jol_job1  -it --rm -u `id -u $USER` -v /sybig/projects/FedL/data:/data -v /sybig/home/jol/Code/HIPT:/mnt jol_hipt python3 /mnt/tissue_classification.py; exec bash'
screen -dmS binary_tissue_classification_lr_decay sh -c 'docker run --shm-size=400gb --gpus \"device=0\" --name jol_job1  -it --rm -u `id -u $USER` -v /sybig/projects/FedL/data:/data -v /sybig/home/jol/Code/HIPT:/mnt jol_hipt python3 /mnt/tissue_classification.py; exec bash'
screen -dmS binary_tissue_classification_lr_decay_patch_based_train_test_split sh -c 'docker run --shm-size=400gb --gpus \"device=0\" --name jol_job1  -it --rm -u `id -u $USER` -v /sybig/projects/FedL/data:/data -v /sybig/home/jol/Code/HIPT:/mnt jol_hipt python3 /mnt/tissue_classification.py; exec bash'
"""

LABEL_MAPPING = {
    1: 0,
    4: 1,
    16: 2,
    17: 3,
    18: 4,
    19: 5,
    20: 6,
    21: 7,
    22: 8,
    23: 9,
    24: 10,
    27: 11,
    28: 12,
    30: 13,
    31: 14,
    32: 15,
    33: 16,
    34: 17,
    36: 18,
    37: 19,
}

REVERSE_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}

BINARY_LABEL_MAPPING = {
    1: 0,
    4: 0,
    16: 1,
    17: 1,
    18: 1,
    19: 0,
    20: 0,
    21: 0,
    22: 0,
    23: 0,
    24: 0,
    27: 0,
    28: 0,
    30: 0,
    31: 0,
    32: 0,
    33: 0,
    34: 0,
    36: 0,
    37: 0,
}


class CustomLRScheduler(LambdaLR):
    def __init__(self, optimizer, total_epochs, decay_rate, last_epoch=-1):
        self.total_epochs = total_epochs
        self.decay_rate = decay_rate
        super().__init__(optimizer, lr_lambda=self.custom_lr_lambda, last_epoch=last_epoch)

    def custom_lr_lambda(self, epoch):
        return (1 + 10 * epoch / self.total_epochs) ** (-self.decay_rate)


class PathDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        data, label = torch.load(self.paths[idx], map_location="cpu")
        return data, torch.tensor(BINARY_LABEL_MAPPING[label.item()])
        # return data, torch.tensor(LABEL_MAPPING[label.item()])


class UNI_classifier(torch.nn.Module):
    def __init__(self, n_classes=20):
        super().__init__()
        self.fc1 = torch.nn.Linear(1024, 1)
        # self.batch_norm = torch.nn.BatchNorm1d(512)
        # self.fc2 = torch.nn.Linear(512, 1)
        # self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        x = self.fc1(x)
        # x = torch.nn.functional.relu(x)
        # x = self.dropout(x)
        # x = self.fc2(x)
        return F.sigmoid(x)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier = UNI_classifier(n_classes=2).to(device)
    # classifier = UNI_classifier(n_classes=20).to(device)
    batch_size = 64
    epochs = 50
    lr = 0.001
    train_losses = []
    test_losses = []

    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CustomLRScheduler(optimizer, total_epochs=epochs, decay_rate=0.76)
    folders = os.listdir("/data/pamly/embeddings")
    random.shuffle(folders)
    # all_paths = []
    # for folder in folders:
    #     for patch in os.listdir(f"/data/pamly/embeddings/{folder}"):
    #         all_paths.append(f"/data/pamly/embeddings/{folder}/{patch}")
    # random.shuffle(all_paths)
    # train_paths = all_paths[:int(0.8 * len(all_paths))]
    # test_paths = all_paths[int(0.8 * len(all_paths)):]
    train_folders = folders[:int(0.8 * len(folders))]
    test_folders = folders[int(0.8 * len(folders)):]
    train_paths = []
    for folder in train_folders:
        for patch in os.listdir(f"/data/pamly/embeddings/{folder}"):
            train_paths.append(f"/data/pamly/embeddings/{folder}/{patch}")
    test_paths = []
    for folder in test_folders:
        for patch in os.listdir(f"/data/pamly/embeddings/{folder}"):
            test_paths.append(f"/data/pamly/embeddings/{folder}/{patch}")
    print(f"Train samples: {len(train_paths)} Test samples: {len(test_paths)}")

    train_set = PathDataset(train_paths)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, prefetch_factor=2)
    test_set = PathDataset(test_paths)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, prefetch_factor=2)

    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        train_loss = 0
        correct = 0
        total = 0
        classifier.train()
        for (data, label) in tqdm(train_loader, "Training"):
            data = data.to(device)
            label = label.float().to(device)
            optimizer.zero_grad()
            output = classifier(data).squeeze()
            loss = F.binary_cross_entropy(output, label)
            pred = output.round()
            correct += torch.sum(pred == label).item()
            total += len(label)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_losses.append(train_loss / len(train_loader))
        print(f"Train Accuracy: {round(correct / total, 4)}")

        classifier.eval()
        correct = 0
        total = 0
        test_loss = 0
        per_class_correct = {i: 0 for i in range(2)}
        # per_class_correct = {i: 0 for i in range(20)}
        per_class_total = {i: 0 for i in range(2)}
        # per_class_total = {i: 0 for i in range(20)}
        for (data, label) in tqdm(test_loader, "Testing"):
            with torch.no_grad():
                data = data.to(device)
                label = label.float().to(device)
                output = classifier(data).squeeze()
                loss = F.binary_cross_entropy(output, label)
                test_loss += loss.item()
                pred = output.round()
                correct += torch.sum(pred == label).item()
                total += len(label)
                for p, l in zip(pred, label):
                    per_class_total[l.item()] += 1
                    if p == l:
                        per_class_correct[l.item()] += 1
        test_losses.append(test_loss / len(test_loader))

        print(f"Test Accuracy: {round(correct / total, 4)}")
        for i in range(2):
            # for i in range(20):
            print(
                # f"Class {pamly.TileLabel.from_int(REVERSE_LABEL_MAPPING[i]).to_string()} ({i}): "
                f"Class {i}: "
                f"{per_class_correct[i]}/{per_class_total[i]} = "
                f"{round(per_class_correct[i] / per_class_total[i], 4) if per_class_total[i] != 0 else 0}")

        scheduler.step()

    if not os.path.exists("./experiments/tissue_classifier_BCE_smaller_model_AdamW"):
        os.makedirs("./experiments/tissue_classifier_BCE_smaller_model_AdamW", exist_ok=True)
    torch.save(classifier.state_dict(), "./experiments/tissue_classifier_BCE_smaller_model_AdamW/classifier.pt")

    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.legend()
    plt.savefig("./plots/tissue_classifier_BCE_smaller_model_AdamW_loss.png")


if __name__ == '__main__':
    main()
