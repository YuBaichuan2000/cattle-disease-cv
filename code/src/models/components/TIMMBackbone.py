from torch import nn
import timm


class TIMMBackbone(nn.Module):
    def __init__(self, model_name: str = 'efficientnet_b0', pretrained=True, num_classes=3):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)
