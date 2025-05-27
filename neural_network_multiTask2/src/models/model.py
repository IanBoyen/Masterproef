from typing import Optional
from torchvision.models import resnet18
from torch.nn import Module, Sigmoid, Linear, Identity


from torchvision.models._api import Weights


class MultiTaskResNet(Module):
    def __init__(self, base_model, num_classes: int, num_regression: int):
        super().__init__()
        self.base_model = base_model
        self.classification_head = Linear(in_features=512, out_features=num_classes)
        self.regression_head = Linear(in_features=512, out_features=num_regression)
        #self.sigmoid = Sigmoid()
        self.num_classes = num_classes
        self.num_regression = num_regression


    def forward(self, x):
        x = self.base_model(x)
        class_output = self.classification_head(x)
        reg_output = self.regression_head(x)

        #s_class_output = self.sigmoid(class_output) * self.num_classes
        #s_reg_output = self.sigmoid(reg_output) * self.num_regression
        return class_output, reg_output


def get_model(name: str, weights: Optional[Weights]):
    if name == 'resnet18':
        base_model = resnet18(weights=weights)
        base_model.fc = Identity()  # Keep intermediate layer
        model = MultiTaskResNet(base_model, num_classes=8, num_regression=1)
    else:
        raise ValueError(f'Unsupported model "{name}"')

    return model