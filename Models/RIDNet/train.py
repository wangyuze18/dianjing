import math
import os.path

from tqdm import tqdm

from dataset import DiffDataset
from augment import *
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from model import RIDNet
from torch import nn
import numpy as np

epochs = 1000
batch_size = 32
patch_size = 80
train_path = os.path.join('.', 'data', 'train')
save_path = os.path.join('.', 'models', 'ridnet.pth')
device = "cuda" if torch.cuda.is_available() else "cpu"
extension = 64


def seed_everything(seed: int = 1314):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore


def train():
    train_tfm = transforms.Compose(
        [

            RandomRotation(),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomCrop(size=patch_size),
            ToTensor()
        ]
    )
    train_set = DiffDataset(path=train_path, transform=train_tfm, ex=extension)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    model = RIDNet(in_channels=1, out_channels=1, num_features=64).to(device)
    criterion = nn.L1Loss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epochs / 2], gamma=0.5)

    best_loss, early_stop_count = math.inf, 0
    for epoch in range(epochs):
        model.train()
        loss_record = []
        train_pbar = tqdm(train_loader, position=0, leave=True)
        for data in train_pbar:
            noisy = data['noisy'].to(device)
            clean = data['clean'].to(device)
            optimizer.zero_grad()
            pred = model(noisy)
            loss = criterion(pred, clean)
            loss.backward()
            optimizer.step()
            train_pbar.set_description(f'Epoch[{epoch + 1}/{epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})
            loss_record.append(loss.detach().item())

        scheduler.step()
        train_loss = sum(loss_record) / len(loss_record)
        print(f'Epoch [{epoch + 1}/{epochs}]: Train loss: {train_loss:.4f}')
        if train_loss < best_loss and epoch > 100:
            best_loss = train_loss
            torch.save(model.state_dict(), save_path)  # Save your best model
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= 400:
            return


if __name__ == '__main__':
    seed_everything(325)
    train()
