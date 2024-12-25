import torch
import torch.nn as nn

from dataloader import CriteoDataset
from torch.utils.data import DataLoader

class FeedForward(nn.Module):
    def __init__(self, num_layers, embed_dim, widening_factor):
        super(FeedForward, self).__init__()
        dim = embed_dim * widening_factor

        self.layers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(embed_dim, dim),
                nn.SiLU(),
            )
            for _ in range(num_layers)
        )
        self.layers.append(nn.Linear(dim, embed_dim))

    def forward(self, x):
        for l in self.layers:
            x = l(x)

        return x


class FeedForward_2(nn.Module):
    def __init__(self, embed_dim, widening_factor=4):
        super(FeedForward_2, self).__init__()

        dim = embed_dim * widening_factor

        self.l1 = nn.Linear(embed_dim, dim, bias=False)
        self.l2 = nn.Linear(embed_dim, dim, bias=False)
        self.silu = nn.SiLU()
        self.final = nn.Linear(dim, embed_dim, bias=False)

    def forward(self, x):
        x1 = self.l1(x)
        x2 = self.l2(x)
        x = self.silu(x1) * x2

        return self.final(x)


class TransformerBlock(nn.Module):
    def __init__(self, num_heads, embed_dim, num_ff_layers=2, widening_factor=4):
        super(TransformerBlock, self).__init__()
        # TODO: assert num heads and embedding dim line up
        # assert
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.0,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # self.ff = FeedForward(num_ff_layers, embed_dim, 4)
        self.ff = FeedForward_2(embed_dim, 4)

    #
    def forward(self, x):
        h = torch.transpose(self.norm1(x), 0, 1)
        h, _ = self.attention(h, h, h)
        x = torch.transpose(h, 0, 1) + x

        x = self.norm2(x)
        h = self.ff(x)
        x = h + x

        return x


class RecommenderTransformer(nn.Module):
    def __init__(
        self, feature_sizes, num_transformer_blocks, num_heads, embed_dim, num_ff_layers
    ):
        super(RecommenderTransformer, self).__init__()
        self.feature_sizes = feature_sizes
        self.embeddings = nn.ModuleList(
            nn.Embedding(
                num_embeddings=field_size + 1,
                embedding_dim=embed_dim,
            )
            for field_size in self.feature_sizes
        )
        self.transformer_blocks = nn.ModuleList(
            TransformerBlock(
                num_heads=num_heads,
                embed_dim=embed_dim,
                num_ff_layers=num_ff_layers,
                widening_factor=4,
            )
            for _ in range(num_transformer_blocks)
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.final_layer = nn.Linear(embed_dim, 1)

    def forward(self, x):
        # Positional encoding not neccesary since features do not contain relative positional information

        # TODO: Find smart way to encode each sequence correctly
        x = torch.stack([
            embed_layer(x[:, i].long())
            for i, embed_layer in enumerate(self.embeddings)
        ], dim=1)  # (B, T) > (B, T, D)
        for l in self.transformer_blocks:
            x = l(x)
        x = x[:, -1, :]  # (B, T, D) > (B, D)
        x = self.norm(x)
        x = self.final_layer(x)  # (B, D) > (B, 1)

        return x

if __name__ == "__main__":
    torch.manual_seed(1246211)

    bs = 2
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
    net = RecommenderTransformer(
        feature_sizes=list(train_dset.field_dims),
        num_transformer_blocks=4,
        num_heads=4,
        embed_dim=k,
        num_ff_layers=2,
    )
    for i, (inputs, labels) in enumerate(train_loader):
        inputs[inputs == float("Inf")] = 0
        logits = net(inputs)
        loss = torch.binary_cross_entropy_with_logits(logits, labels)
        print(loss.item())