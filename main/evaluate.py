from torch.autograd import Variable
from torchvision.transforms import transforms
import cv2
from PIL import Image, ImageFile
import numpy as np
import torch
import sys
sys.path.append('D:\\study\\Code\\python_projects\\MRS\\main')
print(sys.path)

checkpoint = torch.load('./main/resnet-18.t7', map_location=torch.device('cpu'))
net = checkpoint['net']
ImageFile.LOAD_TRUNCATED_IMAGES = True

k1 = 25
k2 = 4
x = 267
left = [x, x - k1, x - 2 * k1 - k2, x - 3 * k1 - k2, x - 4 * k1 - k2]


def getFourNumber(path):
    img = cv2.imread(path)
    numbers = []
    for i in range(4):
        number = img[100: 152, left[i + 1]: left[i] + 3]
        numbers.append(number)
    return img, numbers


def predict(path):
    img, numbers = getFourNumber(path)
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    ans = []

    with torch.no_grad():
        for number in numbers:
            number = Image.fromarray(np.uint8(number))
            number = number.resize((32, 52), Image.ANTIALIAS)
            number = transform_train(number)
            number = np.array(number)
            number = torch.from_numpy(number).float()
            number = Variable(number)
            number = number.unsqueeze(0)
            outputs = net(number)
            predicted = torch.argmax(outputs.data, 1)
            ans.append(predicted.item())
    ans.reverse()
    return ans


if __name__ == '__main__':
    ans = predict('..\\main\\static\\test\\eval.jpg')
    print(ans)
