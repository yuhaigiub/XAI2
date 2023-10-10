import util
import torch
from torch import nn
from nbeatODE.layer import CombinedModel
import torch.optim as optim
class Model():
    def __init__(self, scaler, num_nodes, lrate, wdecay, device, adj_mx, batch_size, timestep, channels, hidden, num_layer):
        self.model = CombinedModel(batch_size, timestep, num_nodes, channels, hidden, num_layer, adj_mx)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate,
                                    weight_decay=wdecay)
        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 5



    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
    
        output = self.model(input)

      
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)

        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return loss.item(), mape, rmse

    def eval(self, input, real_val):
        self.model.eval()
        
        output = self.model(input)
      
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return loss.item(), mape, rmse
