import torch
from dataset import MyDataset
from model import UACNet
from torchinfo import summary
from torch.utils.data import DataLoader
from train_process import train_process
from test_process import test_process
import torch.nn as nn

'''Super parameters''' 
EPOCHS = 100
learning_rate = 0.0001
START_EPOCH = 0
# ======================================================================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # device selection
'''Load dataset'''
train_loader = DataLoader(MyDataset(r'./data/'), batch_size=1, shuffle=True)
test_loader = DataLoader(MyDataset(r'./data/', train=False), batch_size=1, shuffle=False)

uac_net = UACNet().to(DEVICE)
# optimizer = torch.optim.SGD(uac_net.parameters(), lr=learning_rate, weight_decay=1)
optimizer = torch.optim.SGD(uac_net.parameters(), lr=learning_rate, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.1, last_epoch=-1)
CELoss = nn.CrossEntropyLoss()
# print(summary(uac_net, input_size=(1, 1, 6000)))
print(f'learning_rate={learning_rate}')
if START_EPOCH != 0:
    uac_net.load_state_dict(torch.load('./results/checkpoints/UACNet-epoch-' + str(START_EPOCH).zfill(3) + '.pkl'))
# uac_net.eval()
with open('./results/test_results/train_losses.txt', 'w') as f:
    f.write('')
for epoch in range(START_EPOCH, EPOCHS):
    train_process(uac_net, DEVICE, train_loader, optimizer, epoch, CELoss)
    scheduler.step()
    if (epoch+1) % 1 == 0:
        test_process(uac_net, DEVICE, test_loader, epoch, CELoss)


