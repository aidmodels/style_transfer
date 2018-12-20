import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from cvpm.solver import Solver
from cvpm.utility import load_image_file
from style_transfer.bundle import StyleTransferBundle as Bundle
from style_transfer.transformerNet import TransformerNet
import re

class NeuralStyleSolver(Solver):
    def __init__(self, toml=None):
        super().__init__(Bundle.PRETRAINED_TOML)
        self.set_bundle(Bundle)
        self.set_ready()

    def infer(self, image_file, config):
        image_np = self._load_image(image_file)
        raw_results = ""
        if config["style"] == "candy":
            raw_results = Bundle.CANDY_STYLE_LOCATION
        elif config["style"] == "mosaic":
            raw_results = Bundle.MOSAIC_STYLE_LOCATION
        elif config["style"] == "rain_princess":
            raw_results = Bundle.RAIN_PRINCESS_STYLE_LOCATION
        else:
            raw_results = Bundle.UDNIE_STYLE_LOCATION
        
        content_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        content_image = image_np
        content_image = content_transform(content_image)
        content_image = content_image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            style_model = TransformerNet()
            state_dict = torch.load(raw_results)
            for k in list(state_dict.keys()):
                if re.search(r'in\d+\.running_(mean|var)$', k):
                    del state_dict[k]
            style_model.load_state_dict(state_dict)
            style_model.to(device)
            output = style_model(content_image).cpu()
            self._save_image("test.png", output[0])

    def _save_image(self, filename, data):
        img = data.clone().clamp(0, 255).numpy()
        img = img.transpose(1, 2, 0).astype("uint8")
        img = Image.fromarray(img)
        img.save(filename)

    def _load_image(self, filename, size=None, scale=None):
        img = Image.open(filename)
        if size is not None:
            img = img.resize((size, size), Image.ANTIALIAS)
        elif scale is not None:
            img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
        return img