# Altered model from https://github.com/deboradum/deepFM
# Converted from tinygrad to torch because tinygrad .cat can only handle 32 items maximum
# Also changed from one hot encoded vecs to indices.
import torch
import torch.nn as nn


# https://arxiv.org/abs/1703.04247
class DeepNet(nn.Module):
    def __init__(self, input_dim, num_layers, hidden_dim, device):
        super(DeepNet, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(),
            nn.ReLU(),
        ).to(device)
        nn.init.kaiming_uniform_(self.l1[0].weight, mode="fan_in", nonlinearity="relu")

        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(),
                    nn.ReLU(),
                ).to(device)
                for _ in range(num_layers)
            ]
        )
        for layer in self.layers:
            nn.init.kaiming_uniform_(layer[0].weight, mode='fan_in', nonlinearity='relu')

        self.final_layer = nn.Linear(hidden_dim, 1).to(device)
        nn.init.kaiming_uniform_(
            self.final_layer.weight, mode="fan_in", nonlinearity="relu"
        )

    def __call__(self, x):
        x = self.l1(x)
        for layer in self.layers:
            x = layer(x)

        return self.final_layer(x)


class DeepFM(nn.Module):
    # m: Number of feature fields
    # k: embedding size per feature field
    def __init__(self, feature_sizes, k, num_hidden_layers, hidden_dim, device):
        super(DeepFM, self).__init__()
        m = len(feature_sizes)
        self.feature_sizes = feature_sizes

        # +1 because I map inf to 0.
        self.o1_embeddings = [
            nn.Embedding(int(field_size) + 1, 1).to(device)
            for field_size in self.feature_sizes
        ]
        self.o2_embeddings = [
            nn.Embedding(int(field_size) + 1, k).to(device)
            for field_size in self.feature_sizes
        ]

        self.bias = nn.Parameter(torch.Tensor(1).to(device))

        self.deep = DeepNet(
            input_dim=m * k,
            num_layers=num_hidden_layers,
            hidden_dim=hidden_dim,
            device=device,
        ).to(device)

    def __call__(self, x):
        dense_x_o2_list = [
            embed_layer(x[:, i].long())
            for i, embed_layer in enumerate(self.o2_embeddings)
        ]
        dense_x_o2 = torch.cat(dense_x_o2_list, dim=1)

        y_fm = self.fm(x, dense_x_o2_list)
        y_dnn = self.deep(dense_x_o2).sum(dim=1)

        return y_fm + y_dnn

    # https://www.ismll.uni-hildesheim.de/pub/pdfs/Rendle2010FM.pdf
    def fm(self, x, dense_x_o2_list):
        # First order part of FM
        # \sum^{n}_{i=1} w_{i} x_{i}
        dense_x_o1_list = [
            embed_layer(x[:, i].long()) for i, embed_layer in enumerate(self.o1_embeddings)
        ]
        fm_o1 = torch.cat(dense_x_o1_list, dim=1).sum(axis=1)

        # Second order part of FM
        # 0.5 * \sum^{k}_{f=1}( (\sum^{n}_{i=1} v_{i, f} x_{i} )^{2} - \sum^{n}_{i=1} v^{2}_{i, f} x^{2}_{i} )
        # t1 = (\sum^{n}_{i=1} v_{i, f} x_{i} )^{2}
        # t2 = \sum^{n}_{i=1} v^{2}_{i, f} x^{2}_{i}
        t1 = sum(dense_x_o2_list).square()
        t2 = sum([e * e for e in dense_x_o2_list])
        fm_o2 = 0.5 * (t1 - t2).sum(axis=1)

        return self.bias + fm_o1 + fm_o2
