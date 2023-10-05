import torch
import torch.optim as optim
import util

from patchTST.patchTST import PatchTST


class Model():
    def __init__(self, 
                 scaler, num_nodes, lrate, wdecay, device, adj_mx, 
                 context_window, 
                 target_window, 
                 patch_len, 
                 stride):
        self.patchTST = PatchTST(device, context_window, target_window, patch_len, stride)
        self.patchTST.to(device)
        self.optimizer = optim.Adam(self.patchTST.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 5

        self.edge_index = [[], []]
        self.edge_weight = []

        # The adjacency matrix is converted into an edge_index list
        # in accordance with PyG API
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj_mx.item((i, j)) != 0:
                    self.edge_index[0].append(i)
                    self.edge_index[1].append(j)
                    self.edge_weight.append(adj_mx.item((i, j)))

        self.edge_index = torch.tensor(self.edge_index)
        self.edge_weight = torch.tensor(self.edge_weight)

    def train(self, input: torch.Tensor, real_val):
        self.patchTST.train()
        self.optimizer.zero_grad()
        
        input = input[..., 0]        
        input = input.squeeze(-1)
        output = self.patchTST(input)
        
        real_val = real_val[..., 0]
        real = torch.squeeze(real_val, dim=-1)
        predict = self.scaler.inverse_transform(output)
        
        # real = torch.squeeze(real_val, dim=1)
        # predict = self.scaler.inverse_transform(output)

        loss = self.loss(predict, real, 0.0)
        loss.backward()
        
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.patchTST.parameters(), self.clip)
        self.optimizer.step()
        
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        
        return loss.item(), mape, rmse

    def eval(self, input, real_val):
        self.patchTST.eval()
        
        input = input[..., 0]
        input = input.squeeze(-1)
        output = self.patchTST(input)
        
        real_val = real_val[..., 0]
        real = torch.squeeze(real_val, dim=-1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        
        return loss.item(), mape, rmse
