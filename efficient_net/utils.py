from efficient_net.options import OptionsModelType
from efficientnet_pytorch import EfficientNet
import torch


def load_efficient_net_model(
    path_model: str,
    model_type: OptionsModelType,
    num_classes: int,
):
    """This function loads an efficient net model and returns it.

    Args:
        path_model (str): Path to the efficient net model.
        model_type (OptionsModelType): Type of model.

    Returns:
        nn.Module: Loaded model.
    """
    model = EfficientNet.from_pretrained(model_type)
    model._fc = torch.nn.Linear(model._fc.in_features, num_classes)
    checkpoint = torch.load(path_model)
    model.load_state_dict(checkpoint, strict=False)

    return model
