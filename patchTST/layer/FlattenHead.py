from torch import nn

class FlattenHead(nn.Module):
    def __init__(self,
                 nf, 
                 target_window, 
                 head_dropout=0.):
        super(FlattenHead, self).__init__()
        
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)
    
    def forward(self, X):
        X = self.flatten(X)
        X = self.linear(X)
        X = self.dropout(X)
        
        return X