from enum import Enum
from efficient_net.utils import load_efficient_net_model, OptionsModelType


class OptionsClassifModel(str, Enum):
    EFFICIENT_NET = "EfficientNet classification model architecture (B0...B7)"


class OptionsObjDetectModel(str, Enum):
    YOLO_V7 = "Yolo v7 object detection models."
    YOLO_NAS = "Yolo NAS object detection models."


def get_zoo_model(
    model_class: OptionsClassifModel | OptionsObjDetectModel,
) -> callable:
    if isinstance(model_class, OptionsClassifModel):
        if model_class == OptionsClassifModel.EFFICIENT_NET:
            return {"function": load_efficient_net_model, "model_types": OptionsModelType}
