from dataclasses import dataclass
import os
from typing import Any, Dict, List, Literal, Optional, Tuple
from PIL import Image
import yaml

from .generator import Generator


@dataclass
class TextLabel:
    id: str
    text: str
    color: Tuple[int, int, int]
    bg_color: Tuple[int, int, int]

    @staticmethod
    def from_dict(data: dict, id: Any) -> "TextLabel":
        return TextLabel(
            id=id, text=data["text"], color=data["color"], bg_color=data["bg_color"]
        )


class TextLabelGenerator(Generator):
    def __init__(self):
        pass

    def get(self, idx: int) -> Tuple[Image.Image, Optional[TextLabel]]:
        raise NotImplementedError

    @staticmethod
    def _load_images(img_dir: str, convert_mode: str) -> List[Tuple[str, Image.Image]]:
        """
        遍历图像目录，加载所有图像文件。
        """
        if not os.path.isdir(img_dir):
            raise ValueError(f"{img_dir} is not a valid directory")

        images = []
        for filename in os.listdir(img_dir):
            if filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                image_path = os.path.join(img_dir, filename)
                try:
                    with Image.open(image_path).convert(convert_mode) as img:
                        images.append((filename, img.copy()))
                except IOError:
                    print(f"Error opening image file {image_path}")
        return images

    @staticmethod
    def _load_labels(label_path: str) -> Dict[str, TextLabel]:
        """
        从标签文件加载标签。
        """
        assert label_path.endswith(".yaml") or label_path.endswith(
            ".yml"
        ), "Only YAML files are supported"

        with open(label_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        labels = {str(i): TextLabel().from_dict(data[i], i) for i in data}
        return labels


class RawFileGenerator(TextLabelGenerator):
    def __init__(
        self,
        img_dir: str,
        label_path: Optional[str],
        convert_mode: str,
        mode: Literal["train", "eval", "pred"] = "train",
    ):
        self.images = self._load_images(img_dir, convert_mode)
        self.labels = self._load_labels(label_path) if label_path else {}
        self.mode = mode
        self.refine_dataset()

    def __len__(self):
        return len(self.images)

    def refine_dataset(self):
        """
        从数据集中删除无效的图像和标签。
        """
        if self.mode == "pred":
            return

        valid_ids = set(self.labels.keys())
        self.images = [(i, img) for i, img in self.images if i in valid_ids]

    def get(self, idx: int) -> Tuple[Image.Image, Optional[TextLabel]]:
        idx = idx % len(self)
        img_id, img = self.images[idx]
        label = self.labels[img_id] if self.mode != "pred" else None
        return img, label
