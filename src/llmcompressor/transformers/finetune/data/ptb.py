from copy import deepcopy
from typing import TYPE_CHECKING

from llmcompressor.transformers.finetune.data import TextGenerationDataset
from llmcompressor.typing import Processor

if TYPE_CHECKING:
    from llmcompressor.args import DatasetArguments


@TextGenerationDataset.register(name="ptb")
class PtbDataset(TextGenerationDataset):
    """
    Child text generation class for the PTB dataset

    :param dataset_args: configuration settings for dataset loading
    :param split: split from dataset to load, for instance `test` or `train[:5%]`
    :param processor: processor or tokenizer to use on dataset
    """

    def __init__(
        self, dataset_args: "DatasetArguments", split: str, processor: Processor
    ):
        dataset_args = deepcopy(dataset_args)
        dataset_args.dataset = "ptb_text_only"
        dataset_args.text_column = "sentence"

        super().__init__(
            dataset_args=dataset_args,
            split=split,
            processor=processor,
        )
