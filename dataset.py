#%%
import torch
import numpy as np
from PIL import Image
import numpy as np
from torchvision import transforms
from config import *
from utils import *
import cv2

# small to big
class balloons(torch.utils.data.Dataset):
    def __init__(self):
        img = Image.open(image_path_balloon)
        img = transforms.ToTensor()(img).unsqueeze_(0)
        self.imgs = []
        img_size = np.shape(img)

        for i in range(param.num_scale):
            new_h = int(img_size[2] * pow(param.scale_ratio, i))
            new_w = int(img_size[3] * pow(param.scale_ratio, i))

            img_temp = upscale_img(img, (new_h, new_w))

            img_temp = img_norm(img_temp).to(device)
            self.imgs.append(img_temp[0])
        self.imgs.reverse()

    def __getitem__(self, index):
        return self.imgs

    def __len__(self):
        return 1


train_dataset = balloons()
test_dataset = balloons()

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=param.batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=param.batch_size)

