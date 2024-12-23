import os
import json
import time
import torch

from Model import DeepFM
from dataloader import CriteoDataset

from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
device = torch.device(
    "mps"
    if torch.backends.mps.is_built()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def train(net, optimizer, criterion, acc_fn, epochs, train_loader, val_loader, test_loader):
    for e in range(epochs):
        running_loss = 0.
        last_loss = 0.
        accs = []
        s = time.time()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs[inputs == float("Inf")] = 0

            optimizer.zero_grad()
            logits = net(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            accs.append(acc_fn(torch.sigmoid(logits), labels))

            running_loss += loss.item()
            if i % 100 == 99:
                last_loss = running_loss / 100  # loss per batch
                taken = round(time.time()-s, 3)
                tb_x = e * len(train_loader) + i + 1
                mean_acc = torch.Tensor(accs).mean()
                print(f"iter {tb_x} | loss: {last_loss} | acc: {mean_acc} | Took {taken}s")
                writer.add_scalar("Loss/train", last_loss, tb_x)
                running_loss = 0.0
                accs = []
                s = time.time()


def accuracy(predictions, truth):
    return (torch.where(predictions > 0.5, 1, 0) == truth).float().mean().item()


if __name__ == "__main__":
    bs = 1024
    train_dset = CriteoDataset("dataset/train.txt")
    train_loader = DataLoader(train_dset, batch_size=bs, shuffle=True)
    val_loader = DataLoader(CriteoDataset("dataset/val.txt"), batch_size=bs, shuffle=True)
    test_loader = DataLoader(CriteoDataset("dataset/test.txt"), batch_size=bs, shuffle=True)

    net = DeepFM(list(train_dset.field_dims), 64, device).to(device)
    lr = 0.0005
    optimizer = Adam(net.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    train(
        net=net,
        optimizer=optimizer,
        criterion=criterion,
        acc_fn=accuracy,
        epochs=3,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )
