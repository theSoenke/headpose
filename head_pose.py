import argparse

import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from hopenet import Hopenet
from PIL import Image
from torchvision import transforms
from visualization import draw_pose


class HeadPose():
    def __init__(self, checkpoint, transform=None):
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        num_bins = 66
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.idx_tensor = torch.FloatTensor([idx for idx in range(num_bins)]).to(self.device)
        self.model = Hopenet()
        checkpoint = torch.load(checkpoint)
        self.model.load_state_dict(checkpoint, strict=False)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, image):
        if isinstance(image, torch.Tensor) and len(image.shape) == 4:
            data = torch.stack([self.transform(transforms.functional.to_pil_image(img)) for img in image])
        if isinstance(image, torch.Tensor):
            data = self.transform(transforms.functional.to_pil_image(image))
        elif isinstance(image, str):
            image = Image.open(image)
            data = self.transform(image).unsqueeze(dim=0)
        else:
            data = self.transform(image).unsqueeze(dim=0)

        data = data.to(self.device)
        yaw, pitch, roll = self.model(data)
        yaw = F.softmax(yaw, dim=1)
        pitch = F.softmax(pitch, dim=1)
        roll = F.softmax(roll, dim=1)

        yaw = torch.sum(yaw * self.idx_tensor, dim=1) * 3 - 99
        pitch = torch.sum(pitch * self.idx_tensor, dim=1) * 3 - 99
        roll = torch.sum(roll * self.idx_tensor, dim=1) * 3 - 99
        return {'yaw': yaw, 'pitch': pitch, 'roll': roll}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='data/hopenet_robust_alpha1.pkl', type=str)
    parser.add_argument('--image', type=str)
    args = parser.parse_args()

    head_pose = HeadPose(checkpoint=args.checkpoint)
    angles = head_pose.predict(args.image)
    yaw, pitch, roll = angles['yaw'].item(), angles['pitch'].item(), angles['roll'].item()

    print("Yaw: %f" % yaw)
    img = cv2.imread(args.image)
    img = draw_pose(img, yaw, pitch, roll, tdx=200, tdy=200, size=100)
    plt.imshow(img)
    plt.show()
