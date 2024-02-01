import torch
import torch.nn as nn


class biLSTM_TextCNN(nn.Module):
    def __init__(self, embeddings_dim=1024, output_dim=1, dropout=0.25, kernel_size = 9 ,conv_dropout: float = 0.25):
        super(biLSTM_TextCNN, self).__init__()
        
        hidden_size = 256 
       
        self.lstm = nn.LSTM(embeddings_dim, hidden_size, bidirectional=True, batch_first=True)
        
        num_channels = [512,512,512]
        kernel_sizes = [9,6,3]
        self.dropout = nn.Dropout(0.25)
    
        
        self.convs = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(embeddings_dim//2, c, k))
            
        self.decoder = nn.Linear(sum(num_channels), 1)
        self.softmax = nn.Sigmoid()

    def forward(self, x: torch.Tensor, mask, **kwargs) -> torch.Tensor:
        """
        Args:
            x: [batch_size, embeddings_dim, sequence_length] embedding tensor that should be classified
            mask: [batch_size, sequence_length] mask corresponding to the zero padding used for the shorter sequecnes in the batch. All values corresponding to padding are False and the rest is True.

        Returns:
            classification: [batch_size,output_dim] tensor with logits
        """
        # print('x',x.shape)
        x = x.permute(0, 2, 1)
        
        lstm_output, _ = self.lstm(x)
        # print('lstm_output',lstm_output.shape)
       
        lstm_output = lstm_output.permute(0, 2, 1)
        
        cnn_outputs = [conv(lstm_output) for conv in self.convs]
        
        pooled_outputs = [torch.max(cnn_output, dim=2)[0] for cnn_output in cnn_outputs]
        # [batch_size,num_channels]
        
        combined_features = torch.cat(pooled_outputs, dim=1)
        outputs = self.softmax(self.decoder(self.dropout(combined_features)))
        
        return outputs
        
    
    