import torch
from torchvision import transforms as T
from PIL import Image
from basic_model import Net
from GAN_model import Generator
import numpy as np
import os


transform = T.Compose([T.ToTensor(), T.Resize((256, 256)), T.Normalize(0.5, 0.5)])

image = Image.open('picture/0_gray.jpg')
size = image.size
size = (size[1], size[0])
# print(size)
image = transform(image)
image = image.unsqueeze(0)

model = Generator()
model.load_state_dict(torch.load('trained_model/GAN_G2022919214330.pth'))
z = torch.rand(1, 1, 8, 8)
output = model(image, z)
# print(output.shape)
transform = T.Compose([T.Resize(size), T.ToPILImage()])
output = output.squeeze()
# print(output.shape)
img = transform(output)
# print(img.size)
os.makedirs('output',exist_ok=True)
img.save('output/output.jpg')

# img = np.asarray(img)
# print(img[img>255])

