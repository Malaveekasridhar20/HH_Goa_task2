import logging
from datasets import load_dataset
from typing import Optional, Any

logger = logging.getLogger(__name__)

class DatasetLoader:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.dataset = None

    def load(self, split: Optional[str] = None, streaming: bool = False) -> Any:
        """
        Loads the dataset from Hugging Face.
        Optionally load only a specific split.
        """
        try:
            logger.info(f"Loading dataset: {self.dataset_name} (split={split}, streaming={streaming})")
            if split:
                self.dataset = load_dataset(self.dataset_name, split=split, streaming=streaming)
            else:
                self.dataset = load_dataset(self.dataset_name, streaming=streaming)
            logger.info("Dataset loaded successfully.")
            return self.dataset
        except Exception as e:
            logger.error(f"Failed to load dataset '{self.dataset_name}': {e}")
            raise
