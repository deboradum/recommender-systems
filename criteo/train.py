import os
import json
import time
import torch

from DeepFM import DeepFM
from RecommenderTransformer import RecommenderTransformer
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


def evaluate(net, loader, num_batches, acc_fn):
    with torch.no_grad():
        net.eval()
        eval_accs = 0.0
        eval_losses = 0.0
        for j, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs[inputs == float("Inf")] = 0

            logits = net(inputs)
            loss = criterion(logits.view(-1), labels)

            eval_losses += loss.item()
            eval_accs += acc_fn(torch.sigmoid(logits), labels)
            if j >= num_batches:
                break
        mean_eval_loss = eval_losses / num_batches
        mean_eval_acc = eval_accs / num_batches

    return mean_eval_loss, mean_eval_acc


def train(
    net, optimizer, criterion, acc_fn, epochs, train_loader, val_loader, test_loader
):
    log_freq = 250
    net.train()
    for e in range(epochs):
        running_loss = 0.0
        accs = 0.0
        s = time.time()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs[inputs == float("Inf")] = 0

            optimizer.zero_grad()
            logits = net(inputs)
            loss = criterion(logits.view(-1), labels)
            loss.backward()
            optimizer.step()

            accs += acc_fn(torch.sigmoid(logits), labels)
            running_loss += loss.item()

            if i % log_freq == log_freq - 1:
                mean_eval_loss, mean_eval_acc = evaluate(net, val_loader, 350, acc_fn)
                mean_loss, mean_acc = running_loss / log_freq, accs / log_freq

                taken = round(time.time() - s, 3)
                tb_x = e * len(train_loader) + i + 1
                print(
                    f"iter {tb_x} | train loss: {mean_loss} | train acc: {mean_acc} | eval loss: {mean_eval_loss} | eval acc: {mean_eval_acc} | Took {taken}s"
                )
                writer.add_scalar("Loss/train", mean_loss, tb_x)
                writer.add_scalar("Accuracy/train", mean_acc, tb_x)
                writer.add_scalar("Loss/val", mean_eval_loss, tb_x)
                writer.add_scalar("Accuracy/val", mean_eval_acc, tb_x)
                running_loss = 0.0
                accs = 0.0
                s = time.time()
                net.train()

    return evaluate(net, test_loader, 9999999, acc_fn)


def accuracy(predictions, truth):
    return (torch.where(predictions > 0.5, 1, 0) == truth).float().mean().item()


if __name__ == "__main__":
    torch.manual_seed(1246211)

    bs = 1024
    train_dset = CriteoDataset("dataset/train.txt")
    train_loader = DataLoader(train_dset, batch_size=bs, shuffle=False)
    val_loader = DataLoader(
        CriteoDataset("dataset/val.txt"), batch_size=bs, shuffle=False
    )
    test_loader = DataLoader(
        CriteoDataset("dataset/test.txt"), batch_size=bs, shuffle=False
    )

    k = 128
    num_hidden_layers = 4
    hidden_dim = 4096
    # net = DeepFM(
    #     feature_sizes=list(train_dset.field_dims),
    #     k=k,
    #     num_hidden_layers=num_hidden_layers,
    #     hidden_dim=hidden_dim,
    #     device=device,
    # ).to(device)
    net = RecommenderTransformer(
        feature_sizes=list(train_dset.field_dims),
        num_transformer_blocks=4,
        num_heads=4,
        embed_dim=k,
        num_ff_layers=2,
    ).to(device)

    lr = 0.005
    optimizer = Adam(net.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    mean_eval_loss, mean_eval_acc = train(
        net=net,
        optimizer=optimizer,
        criterion=criterion,
        acc_fn=accuracy,
        epochs=3,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )

    print(mean_eval_acc)
    print(mean_eval_loss)
