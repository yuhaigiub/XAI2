import torch
import torch.optim as optim
from pertubate import apply_gaussian_blur
import util

from patchTST.patchTST import PatchTST
from graphwavenet.graphwavenet import GraphWaveNet
class Model():
    def __init__(self, 
                 scaler, num_nodes, lrate, wdecay, device, adj_mx, 
                 context_window, 
                 target_window, 
                 patch_len, 
                 stride,
                 blackbox_file):
        self.patchTST = PatchTST(device, context_window, target_window, patch_len, stride)
        
        self.blackbox = GraphWaveNet(num_nodes, 2, 1, 12)
        self.blackbox.load_state_dict(torch.load(blackbox_file))
        
        self.patchTST.to(device)
        self.blackbox.to(device)
        
        self.device = device
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
        
        inputTST = input[..., 0]        
        inputTST = inputTST.squeeze(-1)
        saliency = self.patchTST(inputTST)
        
        input = apply_gaussian_blur(self.device, input, saliency.unsqueeze(-1))
        
        inputBlackbox = input.transpose(1, 3).transpose(-3, -1)
        real_val = real_val.transpose(1, 3)[:, 0,: ,: ]
        
        output = self.blackbox(inputBlackbox, self.edge_index, self.edge_weight)
        output = output.transpose(-3, -1)
        
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)

        loss = self.loss(predict, real, 0.0)
        loss.backward()
        
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.patchTST.parameters(), self.clip)
        self.optimizer.step()
        
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        
        return loss.item(), mape, rmse

    def eval(self, input: torch.Tensor, real_val):
        self.patchTST.eval()
        
        inputTST = input[..., 0]        
        inputTST = inputTST.squeeze(-1)
        saliency = self.patchTST(inputTST)
        
        input = apply_gaussian_blur(self.device, input, saliency.unsqueeze(-1))
        
        inputBlackbox = input.transpose(1, 3).transpose(-3, -1)
        real_val = real_val.transpose(1, 3)[:, 0,: ,: ]
        
        output = self.blackbox(inputBlackbox, self.edge_index, self.edge_weight)
        output = output.transpose(-3, -1)
        
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)
        
        loss = self.loss(predict, real, 0.0)
        
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        
        return loss.item(), mape, rmse
