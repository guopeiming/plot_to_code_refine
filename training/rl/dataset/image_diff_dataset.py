import re
import logging
from PIL import Image
from omegaconf import DictConfig
from typing import Optional, override, Dict, List

import datasets
import numpy as np
from verl.utils.dataset.rl_dataset import RLHFDataset
from transformers import PreTrainedTokenizer, ProcessorMixin

logger = logging.getLogger(__name__)


def pil_transform(examples: Dict) -> Dict:
    def load(paths: List[str]):
        return [Image.open(p) for p in paths]

    imgs = examples["images"]
    # batched case
    if isinstance(imgs[0], list):
        examples["images"] = [load(p) for p in imgs]
    else:
        examples["images"] = load(imgs)

    return examples


class ImageDiffDataset(RLHFDataset):
    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
        super().__init__(data_files, tokenizer, config, processor, max_samples)

    @override
    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            # here, we change the original verl code to customize the dataset loading process >>>>>>
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # dataframe = dataframe.map(dataset_map_func)
            # dataframe = dataframe.cast_column("images", datasets.Sequence(datasets.Image()))
            dataframe = dataframe.with_transform(pil_transform)
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        total = len(self.dataframe)
        print(f"dataset len: {len(self.dataframe)}")

        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rngs_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rngs_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.select(indices.tolist())
            print(f"selected {self.max_samples} random samples out of {total}")

        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)

    @override
    def _build_messages(self, example: dict):
        messages: list = example.pop(self.prompt_key)

        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                segments = re.split("(<image>|<video>)", content)
                segments = [item for item in segments if item != ""]
                for segment in segments:
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list

        return messages