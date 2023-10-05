import argparse
import os
import time

import numpy as np
import torch
import util
from graphwavenet.model import Model

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
parser.add_argument('--epochs', type=int, default=5, help='')
parser.add_argument('--print_every', type=int, default=200, help='')
parser.add_argument('--save', type=str, default='store/checkpoint',
                    help='save path')

args = parser.parse_args()


def main():
    device = torch.device(args.device)
    _, _, adj_mx = util.load_adj(args.adjdata)
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    engine = Model(scaler, args.num_nodes, args.learning_rate, args.weight_decay, device, adj_mx)
    
    log_file = open('loss_train_log.txt', 'w')
    save_dir = 'saved_models'  # Directory to save the models
    os.makedirs(save_dir, exist_ok=True)
        
    # if not os.path.exists(args.save):
    #     os.makedirs(args.save)
    print("start training...", flush=True)

    his_loss = []
    val_time = []
    train_time = []


    for i in range(1, args.epochs + 1):
        train_loss = []
        train_mape = []
        train_rmse = []
        # t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x,
                    y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            # trainx = trainx[:, :, :, 0:1]
            # trainx = trainx.unsqueeze(1)
            trainx = trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            metrics = engine.train(trainx, trainy[:, 0, :, :])
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            if iter % args.print_every == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: ' + \
                    '{:.4f}, Train RMSE: {:.4f}'
                print(
                    log.format(iter, train_loss[-1], train_mape[-1],
                                train_rmse[-1]), flush=True)
                

     
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)
        
        log_file.write(f'Epoch {i}, Training Loss: {mtrain_loss:.4f}, Training MAPE: {mtrain_mape:.4f}, Training MAE: {mtrain_rmse:.4f} \n')
        log_file.flush()
        
        print(f'Epoch {i}, Training Loss: {mtrain_loss:.4f}')

  
        print(f'epoch {i} trained')
        print(f'Model saved at epochs {i}')
        model_filename = os.path.join(save_dir, f'G_T_model_{i}.pth')
        torch.save(engine, model_filename)
    print("Training finished")


if __name__ == "__main__":
    main()

