import torch
import cv2
import numpy as np


class HoadonOCR:

    def __init__(self):
        # Init parameters, load model here
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.labels = ['highlands', 'others', 'phuclong', 'starbucks']
        self.modeldir = "resnet18.pth"
        self.model = self.loadModel(modeldir=self.modeldir, classes=self.labels, device=self.device)

    # TODO: implement find label
    def find_label(self, img):
        image = self.transformImage(img,self.device)

        self.model.eval()
        with torch.no_grad():
            output = self.model(image.float())
            _, predicted = torch.max(output, dim=1)
            # print(output, self.labels[int(predicted.cpu().numpy())])

        return self.labels[int(predicted.cpu().numpy())]

    def loadModel(self, modeldir, classes, device):
        # define model
        out_features = len(classes)
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
        model.fc = torch.nn.modules.linear.Linear(in_features=512, out_features=out_features, bias=True)

        #load model from file
        model.load_state_dict(torch.load(modeldir, map_location=device))
        model.to(device)
        return model

    def transformImage(self, image, device):
        img_size = 224
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        img_BGR = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert BGR to RGB
        img_resize = cv2.resize(img_BGR, (img_size, img_size))  # resize
        img_resize = np.array(img_resize, dtype="uint8")  # to ndarray
        image = img_resize / 255  # into 0-1
        image_tensor = torch.tensor(image)  # convert to tensor
        for color in range(3):  # nomalize
            image_tensor[:, :, color] = (image_tensor[:, :, color] - mean[color]) / std[color]

        image_tensor = np.transpose(image_tensor, (2, 0, 1))  # transpose
        image_tensor = image_tensor.unsqueeze(0)  # add dimension
        image_tensor = image_tensor.to(device)
        return image_tensor
