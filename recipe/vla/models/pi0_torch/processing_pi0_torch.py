# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Any, ClassVar, Optional

from transformers import PreTrainedTokenizerBase
from transformers.image_processing_utils import BatchFeature, ImageProcessingMixin
from transformers.processing_utils import ProcessorMixin
from transformers.utils import TensorType


class PI0TorchImageProcessor(ImageProcessingMixin):
    model_input_names: ClassVar[list[str]] = ["pixel_values"]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def preprocess(
        self,
        images: Any,
        return_tensors: Optional[str | TensorType] = None,
        **kwargs: Any,
    ) -> BatchFeature:
        raise NotImplementedError(
            "PI0TorchImageProcessor.preprocess() is a stub. "
            "Implement image preprocessing for PI0 Torch (e.g., resizing/normalization/batching) as needed."
        )

    def __call__(self, images: Any, **kwargs: Any) -> BatchFeature:
        return self.preprocess(images, **kwargs)

class PI0TorchProcessor(ProcessorMixin):
    attributes: ClassVar[list[str]] = ["image_processor", "tokenizer"]
    image_processor_class: str = "AutoImageProcessor"
    tokenizer_class: str = "AutoTokenizer"

    def __init__(
        self,
        image_processor: Optional[ImageProcessingMixin] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ) -> None:
        super().__init__(image_processor, tokenizer)

    def __call__(self, *args: Any, **kwargs: Any) -> BatchFeature:
        raise NotImplementedError(
            "PI0TorchProcessor.__call__() is a stub. "
            "Implement text/image joint preprocessing for PI0 Torch as needed."
        )

if __name__ == "__main__":
    save_dir = "/root/data/pi0_torch"

    img_processor = PI0TorchImageProcessor()
    img_processor.save_pretrained(save_dir)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224", trust_remote_code=True)
    tokenizer.save_pretrained(save_dir)

    processor = PI0TorchProcessor(image_processor=img_processor, tokenizer=tokenizer)
    processor.save_pretrained(save_dir)
