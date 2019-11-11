import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.autograd import Variable as V
from math import ceil
import time
import copy
from torchvision.utils import save_image
import os
import shutil
import argparse

class DoomNet(nn.Module):
    def __init__(self, num_classes):
        super(DoomNet,self).__init__()
        self.relu = nn.ELU(inplace=True)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride = 2, padding = 1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, kernel_size = 3, stride = 2, padding = 1)
        self.bn4 = nn.BatchNorm2d(32)
        #self.fc1 = nn.Linear(32 * 3 * 3, 1024)
        #self.dropout = nn.Dropout(p=0.5)
        self.lstm1 = nn.LSTMCell(32 * 3 * 3, 256)
        self.fc_val = nn.Linear(256, 1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x, state):
        hx1i = state[0]
        cx1i = state[1]

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
   
        x=x.view(x.size(0), 32 * 3 * 3)
        #x=self.dropout(self.relu(self.fc1(x)))
        (hx1o,cx1o)=self.lstm1(x,(hx1i,cx1i))
        v=self.fc_val(hx1o)
        y=self.fc(hx1o)

        state = [hx1o, cx1o]

        return (y, v, state)

    def init_hidden(self, batch_size):
        state=[]
        state.append(torch.zeros(batch_size, 256).cuda())
        state.append(torch.zeros(batch_size, 256).cuda())
        return state


class ICM(nn.Module):
    def __init__(self, num_classes, use_depth, use_optflow):
        super(ICM, self).__init__()
        self.use_depth = use_depth
        self.use_optflow = use_optflow
        self.num_classes = num_classes
        self.relu = nn.ELU(inplace=True)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride = 2, padding = 1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, kernel_size = 3, stride = 2, padding = 1)
        self.bn4 = nn.BatchNorm2d(32)
        self.inverse_fc1 = nn.Linear(288 * 2, 256)
        self.inverse_fc2 = nn.Linear(256, num_classes)
        self.forward_fc1 = nn.Linear(288 + num_classes, 256)
        self.forward_fc2 = nn.Linear(256, 288)
    
        if self.use_depth:
            self.deconv4 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1)
            self.dbn4 = nn.BatchNorm2d(32)
            self.deconv3 = nn.ConvTranspose2d(32, 32, kernel_size = 3, stride = 2, padding = 1)
            self.dbn3 = nn.BatchNorm2d(32)
            self.deconv2 = nn.ConvTranspose2d(32, 32, kernel_size = 3, stride = 2, padding = 1)
            self.dbn2 = nn.BatchNorm2d(32)
            self.deconv1 = nn.ConvTranspose2d(32, 1, kernel_size = 3, stride = 2, padding = 1)
            self.tanh = nn.Tanh()
    
        if self.use_optflow:
            self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1)
            self.dbn4 = nn.BatchNorm2d(32)
            self.deconv3 = nn.ConvTranspose2d(32, 32, kernel_size = 3, stride = 2, padding = 1)
            self.dbn3 = nn.BatchNorm2d(32)
            self.deconv2 = nn.ConvTranspose2d(32, 32, kernel_size = 3, stride = 2, padding = 1)
            self.dbn2 = nn.BatchNorm2d(32)
            self.deconv1 = nn.ConvTranspose2d(32, 2, kernel_size = 3, stride = 2, padding = 1)
            self.tanh = nn.Tanh()

    def forward(self, x1, x2, a):
        
        whole_batch = torch.arange(x1.size(0))
        a_in = torch.zeros(x1.size(0), self.num_classes).cuda()
        a_in[whole_batch, a] = 1.0

        x1 = self.relu(self.bn1(self.conv1(x1)))
        x1 = self.relu(self.bn2(self.conv2(x1)))
        x1 = self.relu(self.bn3(self.conv3(x1)))
        x1 = self.relu(self.bn4(self.conv4(x1)))
        emb1 = x1.view(x1.size(0), 32 * 3 * 3)
        if self.use_depth:
            x1 = self.relu(self.dbn4(self.deconv4(x1, (6, 6))))
            x1 = self.relu(self.dbn3(self.deconv3(x1, (11, 11))))
            x1 = self.relu(self.dbn2(self.deconv2(x1, (21, 21))))
            x1 = self.tanh(self.deconv1(x1, (42, 42)))

        x2 = self.relu(self.bn1(self.conv1(x2)))
        x2 = self.relu(self.bn2(self.conv2(x2)))
        x2 = self.relu(self.bn3(self.conv3(x2)))
        x2 = self.relu(self.bn4(self.conv4(x2)))
        emb2 = x2.view(x2.size(0), 32 * 3 * 3)
        if self.use_depth:
            x2 = self.relu(self.dbn4(self.deconv4(x2, (6, 6))))
            x2 = self.relu(self.dbn3(self.deconv3(x2, (11, 11))))
            x2 = self.relu(self.dbn2(self.deconv2(x2, (21, 21))))
            x2 = self.tanh(self.deconv1(x2, (42, 42)))

        if self.use_optflow:
            emb = torch.cat((emb1, emb2), 1)
            x1 = emb.view(emb.size(0), 64, 3, 3)
            x1 = self.relu(self.dbn4(self.deconv4(x1, (6, 6))))
            x1 = self.relu(self.dbn3(self.deconv3(x1, (11, 11))))
            x1 = self.relu(self.dbn2(self.deconv2(x1, (21, 21))))
            x1 = self.tanh(self.deconv1(x1, (42, 42)))
        
        a_out = torch.randn(x1.size(0), self.num_classes)
        if self.use_depth == False and self.use_optflow == False:
            x = torch.cat((emb1, emb2), 1)
            x = self.relu(self.inverse_fc1(x))
            a_out = self.inverse_fc2(x)

        x = torch.cat((emb1, a_in), 1)
        x = self.relu(self.forward_fc1(x))
        emb2_out = self.forward_fc2(x)

        return (x1, x2, a_out, emb2_out, emb2.detach())

