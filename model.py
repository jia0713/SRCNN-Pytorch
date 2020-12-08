import os
import h5py
import torch
import torchvision 

import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
# from config import Config

class SRCNN(nn.Module):
    def __init__(self, cfg):
        super(SRCNN, self).__init__()
        self.cfg = cfg
        self.image_size = self.cfg.image_size
        self.label_size = self.cfg.label_size
        self.batch_size = self.cfg.batch_size
        self.conv1 = nn.Conv2d(1, 64, 9, 1)
        self.conv2 = nn.Conv2d(64, 32, 1, 1)
        self.conv3 = nn.Conv2d(32, 1, 5, 1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        return x
    
def train(cfg):
    checkpoint_path = os.path.join(os.getcwd(), "checkpoint")
    train_file_path = os.path.join(checkpoint_path, "train.h5")
    with h5py.File(train_file_path, 'r') as hf:
        data_array = np.array(hf.get('data'))
        label_array = np.array(hf.get('label'))
    use_cuda = cfg.train.use_cuda
    device = torch.device("cuda" if use_cuda else "cpu")
    model = SRCNN(cfg)
    model = model.to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=cfg.train.lr)
    epoches = cfg.train.epoches
    batch_size = cfg.batch_size
    num_of_batch = len(data_array) // batch_size
    for epoch in range(1, epoches + 1):
        loss_array = np.zeros(num_of_batch)
        # print(r"Epoch:{}".format(epoch))
        for idx in range(num_of_batch):
            train_batch, label_batch = data_array[idx*batch_size:(idx+1)*batch_size],\
                label_array[idx*batch_size:(idx+1)*batch_size]
            train_batch, label_batch = torch.FloatTensor(train_batch).to(device), torch.FloatTensor(label_batch).to(device)
            output = model(train_batch)
            loss = F.mse_loss(output, label_batch)
            loss.backward()
            optimizer.step()
            loss_array[idx] = loss.item()
        mean_loss = np.mean(loss_array)
        if epoch % 10 == 0:
            print(r"Epoch:{:d}  Loss:{:.4f}".format(epoch, mean_loss))
        if epoch % 100 == 0:
            torch.save(model.state_dict(), os.path.join(checkpoint_path, \
                "srcnn_ep:{:3d}_loss:{:.4f}.pt".format(epoch, mean_loss)))


# if __name__ == "__main__":
#     cfg = Config()
#     train(cfg)




        


