# Altered model from https://github.com/deboradum/deepFM
# Converted from tinygrad to torch because tinygrad .cat can only handle 32 items maximum
# Also changed from one hot encoded vecs to indices.
import torch
import torch.nn as nn
from RecommenderTransformer import RecommenderTransformer


class TransFM(nn.Module):
    # m: Number of feature fields
    # k: embedding size per feature field
    def __init__(
        self,
        feature_sizes,
        k,
        num_transformer_blocks,
        num_heads,
        widening_factor,
        device,
    ):
        super(TransFM, self).__init__()
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

        self.final_layer = nn.Linear(2, 1)

        self.deep = RecommenderTransformer(
            feature_sizes=feature_sizes,
            num_transformer_blocks=num_transformer_blocks,
            num_heads=num_heads,
            embed_dim=k,
            widening_factor=widening_factor,
        ).to(device)

    def forward(self, x):
        dense_x_o2_list = [
            embed_layer(x[:, i].long())
            for i, embed_layer in enumerate(self.o2_embeddings)
        ]

        y_fm = self.fm(x, dense_x_o2_list)
        y_dnn = self.deep(x)

        return self.final_layer(torch.stack([y_fm, y_dnn], dim=1)).view(-1)

    # https://www.ismll.uni-hildesheim.de/pub/pdfs/Rendle2010FM.pdf
    def fm(self, x, dense_x_o2_list):
        # First order part of FM
        # \sum^{n}_{i=1} w_{i} x_{i}
        dense_x_o1_list = [
            embed_layer(x[:, i].long())
            for i, embed_layer in enumerate(self.o1_embeddings)
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
