from enum import Enum


class OptionsModelType(str, Enum):
    B0 = "efficientnet-b0"
    B1 = "efficientnet-b1"
    B2 = "efficientnet-b2"
    B3 = "efficientnet-b3"
    B4 = "efficientnet-b4"
    B5 = "efficientnet-b5"
    B6 = "efficientnet-b6"
    B7 = "efficientnet-b7"

    def get_resolution(self) -> int:
        if self.value == OptionsModelType.B0:
            return 224
        elif self.value == OptionsModelType.B1:
            return 240
        elif self.value == OptionsModelType.B2:
            return 260
        elif self.value == OptionsModelType.B3:
            return 300
        elif self.value == OptionsModelType.B4:
            return 380
        elif self.value == OptionsModelType.B5:
            return 456
        elif self.value == OptionsModelType.B6:
            return 528
        elif self.value == OptionsModelType.B7:
            return 600
        else:
            raise ValueError("Invalid model type")
