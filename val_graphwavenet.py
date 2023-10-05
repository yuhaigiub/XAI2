import argparse
import os
import time

import numpy as np
import torch
import util


parser = argparse.ArgumentParser()



parser.add_argument('--device', type=str, default='cuda:0',
                    help='device to run the model on')
parser.add_argument('--data', type=str, default='store/METR-LA',
                    help='data path')
parser.add_argument('--adjdata', type=str, default='store/adj_mx.pkl',
                    help='adj data path')
parser.add_argument('--in_dim', type=int, default=2, help='inputs dimension')
parser.add_argument('--num_nodes', type=int, default=207,
                    help='number of nodes')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.0001,
                    help='weight decay rate')
parser.add_argument('--epochs', type=int, default=1, help='')
parser.add_argument('--print_every', type=int, default=200, help='')
parser.add_argument('--save', type=str, default='store/checkpoint',
                    help='save path')

args = parser.parse_args()


def main():
    device = torch.device(args.device)
    _, _, adj_mx = util.load_adj(args.adjdata)
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    
    log_file = open('loss_val_log.txt', 'w')
   
    
    print("start eval...", flush=True)

    his_loss = []
    val_time = []
    train_time = []
    
    max_pth =5
    for itea in range(max_pth): 
        engine = torch.load('saved_models' + f'/G_T_model_{itea+1}.pth')

        for i in range(1, args.epochs + 1):
            valid_loss = []
            valid_mape = []
            valid_rmse = []

           
            for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
                testx = torch.Tensor(x).to(device)
                # testx = testx[:, :, :, 0:1]
                testx = testx.transpose(1, 3)
                testy = torch.Tensor(y).to(device)
                testy = testy.transpose(1, 3)
                metrics = engine.eval(testx, testy[:, 0, :, :])
                valid_loss.append(metrics[0])
                valid_mape.append(metrics[1])
                valid_rmse.append(metrics[2])
                
            mtrain_loss = np.mean(valid_loss)
            mtrain_mape = np.mean(valid_mape)
            mtrain_rmse = np.mean(valid_rmse)
            
            log_file.write(f'Epoch {itea + 1}, Val Loss: {mtrain_loss:.4f}, Val MAPE: {mtrain_mape:.4f}, Val MAE: {mtrain_rmse:.4f} \n')
            log_file.flush()
            
            print(f'Epoch {itea + 1}, Val Loss: {mtrain_loss:.4f}')

    print("Evaluating finished")

if __name__ == "__main__":
 
    main()

