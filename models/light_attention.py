import torch
import torch.nn as nn


class LightAttention(nn.Module):
    def __init__(self, embeddings_dim=1024, output_dim=1, dropout=0.25, kernel_size=9, conv_dropout: float = 0.25):
        super(LightAttention, self).__init__()

        self.feature_convolution = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1,
                                             padding=kernel_size // 2)
        self.attention_convolution = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1,
                                               padding=kernel_size // 2)

        self.softmax = nn.Softmax(dim=-1)

        self.dropout = nn.Dropout(conv_dropout)

        self.linear = nn.Sequential(
            nn.Linear(2 * embeddings_dim, 32),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )

        self.output = self.output = nn.Sequential(*[
            nn.Linear(32, output_dim),
            nn.Sigmoid()
        ])
        

    def forward(self, x: torch.Tensor, mask, **kwargs) -> torch.Tensor:
        """
        Args:
            x: [batch_size, embeddings_dim, sequence_length] embedding tensor that should be classified
            mask: [batch_size, sequence_length] mask corresponding to the zero padding used for the shorter sequecnes in the batch. All values corresponding to padding are False and the rest is True.

        Returns:
            classification: [batch_size,output_dim] tensor with logits
        """
        # print('x',x.shape)
        o = self.feature_convolution(x)  # [batch_size, embeddings_dim, sequence_length]
        # print('o',o.shape)
        # x torch.Size([100, 1024, 1021])
        # o torch.Size([100, 1024, 1021])
        
        o = self.dropout(o)  # [batch_gsize, embeddings_dim, sequence_length]
        # print('attention_1',o.shape)
        attention = self.attention_convolution(x)  # [batch_size, embeddings_dim, sequence_length]
        # print('attention_2',attention.shape)
        # mask out the padding to which we do not want to pay any attention (we have the padding because the sequences have different lenghts).
        # This padding is added by the dataloader when using the padded_permuted_collate function in utils/general.py
        # print('mask',mask[:, None, :]== False)
        attention = attention.masked_fill(mask[:, None, :] == False, -1e9)
        # print('attention',attention.shape)
        # code used for extracting embeddings for UMAP visualizations
        # extraction =  torch.sum(x * self.softmax(attention), dim=-1)
        # extraction = self.id0(extraction)

        o1 = torch.sum(o * self.softmax(attention), dim=-1)  # [batchsize, embeddings_dim]
        o2, _ = torch.max(o, dim=-1)  # [batchsize, embeddings_dim]
        o = torch.cat([o1, o2], dim=-1)  # [batchsize, 2*embeddings_dim]
        o = self.linear(o)  # [batchsize, 32]
        return self.output(o)  # [batchsize, output_dim]
