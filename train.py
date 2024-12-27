import yaml
import time
import torch
import random
import string
import argparse

from DeepFM import DeepFM
from RecommenderTransformer import RecommenderTransformer
from criteo.dataloader import CriteoDataset
from movielens.dataloader import MovieLens20MDataset
from avazu.dataloader import AvazuDataset

from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def evaluate(net, loader, num_batches, acc_fn):
    with torch.no_grad():
        net.eval()
        eval_accs = 0.0
        eval_losses = 0.0
        for j, (inputs, labels) in enumerate(loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs[inputs == float("Inf")] = 0

            logits = net(inputs)
            loss = criterion(logits, labels)

            eval_losses += loss.item()
            eval_accs += acc_fn(torch.sigmoid(logits), labels)
            if j >= num_batches - 1:
                break
        mean_eval_loss = eval_losses / num_batches
        mean_eval_acc = eval_accs / num_batches

    return mean_eval_loss, mean_eval_acc


def train(
    net,
    optimizer,
    criterion,
    acc_fn,
    epochs,
    train_loader,
    val_loader,
    test_loader,
    patience=10,
    log_freq=250,
    save_path="best_model.pt"
):
    net.train()
    for e in range(epochs):
        running_loss = 0.0
        accs = 0.0
        best_eval_loss = 99999
        early_stopping_counter = 0
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

            accs += acc_fn(torch.sigmoid(logits), labels)
            running_loss += loss.item()

            if i % log_freq == log_freq - 1:
                num_val_batches = min(log_freq, len(val_loader))
                mean_eval_loss, mean_eval_acc = evaluate(
                    net, val_loader, num_val_batches, acc_fn
                )
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

                if mean_eval_loss < best_eval_loss:
                    best_eval_loss = mean_eval_loss
                    early_stopping_counter = 0
                    best_state_dict = net.state_dict()
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= patience:
                        print(f"Early stopping at iteration {tb_x} with eval loss {best_eval_loss}")
                        torch.save(best_state_dict, save_path)
                        best_model = net
                        best_model.load_state_dict(torch.load(save_path, map_location=device))
                        best_model = best_model.to(device)

                        num_test_batches = min(2000, len(test_loader))
                        return evaluate(best_model, test_loader, num_test_batches, acc_fn)

    torch.save(net.state_dict(), save_path)
    num_test_batches = min(2000, len(test_loader))
    return evaluate(net, test_loader, num_test_batches, acc_fn)


def accuracy(predictions, truth):
    return (torch.where(predictions > 0.5, 1, 0) == truth).float().mean().item()


def parse_args():
    parser = argparse.ArgumentParser(description="Parse command-line arguments.")
    parser.add_argument(
        "-n", "--net", type=str, required=True, choices=["deepfm", "transformer"]
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=True,
        choices=["criteo", "movielens", "avazu"],
    )
    parser.add_argument("-c", "--config", type=str, default="config.yaml")

    return parser.parse_args()


def get_net(net, feature_sizes, config):
    if net == "deepfm":
        net = DeepFM(
            feature_sizes=feature_sizes,
            k=config["network"][net]["k"],
            num_hidden_layers=config["network"][net]["num_hidden_layers"],
            hidden_dim=config["network"][net]["hidden_dim"],
            device=device,
        ).to(device)
    elif net == "transformer":
        net = RecommenderTransformer(
            feature_sizes=feature_sizes,
            num_transformer_blocks=config["network"][net]["num_transformer_blocks"],
            num_heads=config["network"][net]["num_heads"],
            embed_dim=config["network"][net]["k"],
            widening_factor=config["network"][net]["widening_factor"],
        )
    else:
        raise NotImplementedError(f"Net {net} not yet supported")

    return net


def get_loaders(dset, bs):
    if dset == "criteo":
        train_loader = DataLoader(
            CriteoDataset("criteo/dataset/train.txt"), batch_size=bs, shuffle=True
        )
        val_loader = DataLoader(
            CriteoDataset("criteo/dataset/val.txt"), batch_size=bs, shuffle=False
        )
        test_loader = DataLoader(
            CriteoDataset("criteo/dataset/test.txt"), batch_size=bs, shuffle=False
        )
    elif dset == "movielens":
        train_loader = DataLoader(
            MovieLens20MDataset("movielens/ml-20m/train.txt"), batch_size=bs, shuffle=True
        )
        val_loader = DataLoader(
            MovieLens20MDataset("movielens/ml-20m/val.txt"), batch_size=bs, shuffle=False
        )
        test_loader = DataLoader(
            MovieLens20MDataset("movielens/ml-20m/test.txt"), batch_size=bs, shuffle=False
        )
    elif dset == "avazu":
        train_loader = DataLoader(
            AvazuDataset("avazu/avazu-ctr-prediction/train.txt"),
            batch_size=bs,
            shuffle=True,
        )
        val_loader = DataLoader(
            AvazuDataset("avazu/avazu-ctr-prediction/val.txt"),
            batch_size=bs,
            shuffle=False,
        )
        test_loader = DataLoader(
            AvazuDataset("avazu/avazu-ctr-prediction/test.txt"),
            batch_size=bs,
            shuffle=False,
        )
    return train_loader, val_loader, test_loader


def get_config(path):
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config


if __name__ == "__main__":
    args = parse_args()
    conf = get_config(args.config)
    train_loader, val_loader, test_loader = get_loaders(
        args.dataset, conf["training"]["bs"]
    )
    net = get_net(args.net, list(train_loader.dataset.field_dims), conf).to(device)

    # Hyper params for in tensor board logging
    hparams = {
        "seed": conf["training"]["seed"],
        "batch_size": conf["training"]["bs"],
        "lr": conf["training"]["lr"],
        "patience": conf["training"]["patience"],
        "network": args.net,
    }
    network_hparams = conf["network"].get(args.net, {})
    hparams.update(
        {f"{args.net}_{key}": value for key, value in network_hparams.items()}
    )
    print("Hyperparameters:", hparams)

    torch.manual_seed(hparams["seed"])
    lr = hparams["lr"]
    optimizer = Adam(net.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()


    random_suffix = "".join(random.choices(string.ascii_letters + string.digits, k=6))
    run_name = f"{args.dataset}_{args.net}_{random_suffix}"
    writer = SummaryWriter(log_dir=f"runs/{run_name}")

    test_loss, test_acc = train(
        net=net,
        optimizer=optimizer,
        criterion=criterion,
        acc_fn=accuracy,
        epochs=conf["training"]["epochs"],
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        patience=hparams["patience"],
        log_freq=conf["training"]["log_freq"],
        save_path="best_model.pt",
    )

    print(f"Test loss: {test_loss} | Test accuracy: {test_acc}")

    writer.add_scalar("Loss/test", test_loss)
    writer.add_scalar("Accuracy/test", test_acc)

    writer.add_hparams(hparams, {"test_loss": test_loss, "test_acc": test_acc})

    writer.close()
