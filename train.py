import numpy as np
import os
import random
import json
import monai

from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from ./Model/Unet import *

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(num_classes=None, device=None, ckpt_dir=None, num_epoch=100, lr=1e-4):

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=6)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=6)

    DiceCELoss = monai.losses.DiceCELoss(include_background=False, softmax=True, to_onehot_y=True)

    net = UNet(num_classes=num_classes)
    net = net.to(device)

    optim = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optim, factor=0.1, patience=5, mode='min')

    best_loss = 100
    early_stopping = EarlyStopping(patience=20, verbose=True, path=ckpt_dir)

    for epoch in range(1, num_epoch + 1):
        print("-" * 10)
        print(f"epoch {epoch}/{num_epoch}")

        net.train()
        loss_arr = []

        for batch, data in enumerate(train_loader):
            image = data['image'].to(device).float()
            label = data['label'].to(device).long()

            output = net(image)

            optim.zero_grad()
            loss = DiceCELoss(output, label)
            loss.backward()
            optim.step()

            loss_arr += [loss.item()]

        print(f"train average loss: {np.mean(loss_arr):.4f}")

        with torch.no_grad():
            net.eval()
            loss_arr = []

            for batch, data in enumerate(val_loader):
                image = data['image'].to(device).float()
                label = data['label'].to(device).long()

                output = net(image)

                loss = DiceCELoss(output, label)
                loss_arr += [loss.item()]

            print(f"Learning Rate: {scheduler.optimizer.state_dict()['param_groups'][0]['lr']}")
            print(f"validation average loss: {np.mean(loss_arr):.4f}")

            early_stopping(np.mean(loss_arr), net, optim, epoch)

            if early_stopping.early_stop:
                print("Early stopping")
                break

        if np.mean(loss_arr) < best_loss:
            torch.save({'net': net.state_dict(), 'optim': optim.state_dict()}, '%s/model_epoch%d.pth' % (ckpt_dir, epoch))
            print('model_save')
            best_loss = np.mean(loss_arr)

        scheduler.step(np.mean(loss_arr))



if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    seed = 42
    seed_everything(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("*************", device, "*****************")

    root_path = '/home/dawun/Mmode/Train'
    data_view = 'Mmode_LV'

    if data_view == 'Mmode_LA_Ao':
        num_classes = 3

    elif data_view == 'Mmode_LV':
        num_classes = 4

    ckpt_dir = os.path.join(root_path, data_view, 'Checkpoint')
    os.makedirs(ckpt_dir, exist_ok=True)

    train(num_classes=num_classes, device=device)
