from dataset import GetDataset
from torch.utils.data import DataLoader
from torchvision import transforms

if __name__ == "__main__":


    train = GetDataset(root_dir='images/train/', train=True)
    test = GetDataset(root_dir='images/validation/', train=False)

    train_loader = DataLoader(train, batch_size=32, shuffle=True)
    test_loader = DataLoader(test, batch_size=32, shuffle=True)


