import torch
import torch.nn.functional as F
import torch.nn as nn
from torchdiffeq import odeint


class ODEFunc(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ODEFunc, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, t, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class ODEBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, time_steps):
        super(ODEBlock, self).__init__()
        self.odefunc = ODEFunc(input_dim, hidden_dim)
        self.time_steps = time_steps

    def forward(self, x):
        t = torch.linspace(0, 1, self.time_steps).to(x.device)
        out = odeint(self.odefunc, x, t, method='euler')
        return out


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GCN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, x, adjacency_matrix):
        out = torch.matmul(adjacency_matrix, x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class STGCNBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, adjacency_matrix):
        super(STGCNBlock, self).__init__()
        self.gcn = GCN(input_dim, hidden_dim)
        self.adjacency_matrix = adjacency_matrix

    def forward(self, x):
        x = self.gcn(x, self.adjacency_matrix)
        return x


class ODEGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, time_steps, adjacency_matrix):
        super(ODEGCN, self).__init__()
        self.ode_block = ODEBlock(input_dim, hidden_dim, time_steps)
        self.stgcn_block = STGCNBlock(input_dim, hidden_dim, adjacency_matrix)

    def forward(self, x):
        x = self.ode_block(x)
        x = self.stgcn_block(x)
        return x[-1]  


class NBEATSBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_units, num_layers):
        super(NBEATSBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, out_channels)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        forecast = self.fc3(x)
        return forecast

class NBEATS(nn.Module):
    def __init__(self, batch_size, timesteps, num_nodes, channels, hidden_units, num_layers):
        super(NBEATS, self).__init__()
        self.batch_size = batch_size
        self.channels = channels
        self.num_nodes = num_nodes
        self.timesteps = timesteps

        self.forecast_blocks = nn.ModuleList([
            NBEATSBlock(in_channels=num_nodes, out_channels=num_nodes, hidden_units=hidden_units, num_layers=num_layers)
            for _ in range(timesteps)
        ])
        self.backcast_blocks = nn.ModuleList([
            NBEATSBlock(in_channels=num_nodes, out_channels=num_nodes, hidden_units=hidden_units, num_layers=num_layers)
            for _ in range(timesteps)
        ])

    def forward(self, x):
        forecast_outputs = []
        backcast_outputs = []
        x = x.permute(0, 3, 2, 1)
        for timestep in range(self.timesteps):
            forecast = self.forecast_blocks[timestep](x[:, :, :, timestep])
            backcast = self.backcast_blocks[timestep](x[:, :, :, timestep])
            forecast_outputs.append(forecast)
            backcast_outputs.append(backcast)

        forecast = torch.stack(forecast_outputs, dim=-1)  
        backcast = torch.stack(backcast_outputs, dim=-1)  # Kết hợp backcast cho tất cả các bước thời gian
        forecast=forecast.permute(0, 3, 2, 1)
        backcast=backcast.permute(0, 3, 2, 1)

        return forecast
    
    
    
class CombinedModel(nn.Module):
    def __init__(self, batch_size, timesteps, num_nodes, channels, hidden_dim, num_layers, adjacency_matrix):
        super(CombinedModel, self).__init__()
        self.ode_block = ODEGCN(channels, hidden_dim, timesteps, adjacency_matrix)
        self.nbeats_block = NBEATS(batch_size, timesteps, num_nodes, channels, hidden_dim, num_layers)

    def forward(self, x):
        variable_ode = self.ode_block(x)
        forecast= self.nbeats_block(variable_ode)
        return forecast


