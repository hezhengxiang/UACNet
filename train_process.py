from tqdm import tqdm
import torch
import numpy as np


def train_process(model, device, train_loader, optimizer, epoch, celoss):
    model.train()
    losses = []
    loop = tqdm(enumerate(train_loader), total=len(train_loader))
    for index, (wave, label) in loop:
        wave, label = wave.to(device), label.to(device)
        optimizer.zero_grad()
        yp = model(wave.float())
        loss = celoss(torch.unsqueeze(yp, 0), label.long())
        lll = loss
        # ccc = loss.grad
        loss.backward()
        optimizer.step()
        losses.append(np.array(lll.item()))
        loop.set_description(f'Epoch [{epoch}]')
        loop.set_postfix(loss=sum(losses)/(index+1))
    with open('./results/test_results/train_losses.txt', 'a+') as f:
        f.write(str(np.mean(losses))+'\t\n')
    torch.save(model.state_dict(), "./results/checkpoints/UACNet-epoch-" + str(epoch).zfill(3) + ".pkl")
