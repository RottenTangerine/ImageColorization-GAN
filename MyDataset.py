from torch.utils.data import Dataset
import torchvision.transforms as T

from PIL import Image

import os

class ColorizeData(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.image_list = os.listdir(file_path)
        self.train_data_transform = T.Compose([T.ToTensor(),
                                       T.Grayscale(),
                                       T.Resize(size=(256, 256)),
                                       T.Normalize(0.5, 0.5)])

        self.target_data_transform = T.Compose([T.ToTensor(),
                                       T.Resize(size=(256, 256)),
                                       T.Normalize(0.5, 0.5)])

    def read_data(self, index):
        img = Image.open(os.path.join(self.file_path, self.image_list[index])).convert('RGB')
        data = self.train_data_transform(img)
        target = self.target_data_transform(img)
        return data, target

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index:int):
        try:
            data, target = self.read_data(index)
        except Exception as e:
            # open default image
            data, target = self.read_data(0)
        return data, target


if __name__ == '__main__':
    from torch.utils.data import DataLoader, random_split
    dataset = ColorizeData('dataset/colored_manga/color_full')
    split_ratio = 0.2
    train_dataset, validate_dataset = random_split(dataset,
                                                   [l := round(len(dataset) * (1 - split_ratio)), len(dataset) - l])
    train_loader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)
    validate_loader = DataLoader(dataset=validate_dataset, batch_size=8, shuffle=True)

