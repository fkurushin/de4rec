from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import dill
import evaluate
import numpy as np
import torch
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import (PretrainedConfig, PreTrainedModel, Trainer,
                          TrainingArguments)
from transformers.modeling_outputs import ModelOutput
from transformers.trainer_utils import EvalPrediction


class ListDataset(torch.utils.data.Dataset):
    """
    List of tuples of uder_id, item_id, label
    """

    def __init__(self, data: list[list[int, int, int]]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __getitems__(self, idx_list):
        return [self.data[_] for _ in idx_list]

    def distinct_size(self) -> int:
        res = set()
        for lst in self.data:
            res.update(lst)
        return len(res)


class TextDataCollatorForList:
    """
    Convert DataList to named tensors and move to device
    """

    def __call__(self, batch):
        text_ids, tokens_ids, labels = [], [], []
        for text_id, tokens_id, label in batch:
            text_ids.append(text_id)
            tokens_ids.append(torch.LongTensor(tokens_id))
            labels.append(label)

        return {
            "text_ids": torch.LongTensor(text_ids),
            "tokens_ids": pad_sequence(
                tokens_ids, batch_first=True, padding_value=0
            ),  ## tuple of tokens of search_text
            "labels": torch.LongTensor(labels),
        }


@dataclass
class TextEncoderOutput(ModelOutput):
    """
    Output of the TextEncoder model
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    labels: Optional[torch.LongTensor] = None


class TextEncoderConfig(PretrainedConfig):
    model_type = "TextEncoder"

    def __init__(
        self,
        tokens_size: int = None,
        texts_size: int = None,
        embedding_dim: int = None,
        margin: float = 0.85,
        **kwargs,
    ):
        self.tokens_size = tokens_size
        self.texts_size = texts_size
        self.embedding_dim = embedding_dim
        self.margin = margin
        super().__init__(**kwargs)


class TextEncoderModel(PreTrainedModel):
    config_class = TextEncoderConfig

    def __init__(self, config: TextEncoderConfig):
        super().__init__(config)

        self.text_embeddings = torch.nn.Embedding(
            config.texts_size, embedding_dim=config.embedding_dim
        )

        self.token_embeddings = torch.nn.Embedding(
            config.tokens_size, embedding_dim=config.embedding_dim, padding_idx=0
        )
        self.cel = torch.nn.CosineEmbeddingLoss(margin=config.margin, reduction="mean")
        self.cs = torch.nn.CosineSimilarity()

    def forward(self, **kwargs) -> TextEncoderOutput:
        token_embs = self.token_embeddings(kwargs["tokens_ids"]).sum(dim=1)
        text_embs = self.text_embeddings(kwargs["text_ids"])
        logits = self.cs(token_embs, text_embs)
        loss = self.cel(token_embs, text_embs, kwargs["labels"])
        return TextEncoderOutput(loss=loss, logits=logits, labels=kwargs["labels"])

    def find_text_by_tokens(self, token_ids: list[int]) -> int:
        """
        Get tuple of token_ids return closest search_text (item_id from DE)
        ---
        Return:
        text_id
        """
        text_id = torch.topk(
            self.cs(
                self.token_embeddings(torch.LongTensor(token_ids))
                .sum(dim=0)
                .to(self.text_embeddings.weight.device),
                self.text_embeddings.weight,
            ).detach(),
            k=1,
        ).indices[0]
        return text_id


class TextEncoderTrainingArguments(TrainingArguments):
    def __init__(self, **kwargs):
        self._kwargs = kwargs
        super().__init__(
            output_dir=kwargs.get("output_dir", "./results"),
            eval_strategy=kwargs.get("eval_strategy", "steps"),
            logging_steps=kwargs.get("logging_steps", 100),
            prediction_loss_only=False,
            save_strategy="best",
            load_best_model_at_end=True,
            metric_for_best_model=kwargs.get("metric_for_best_model", "eval_loss"),
            learning_rate=kwargs.get("learning_rate", 1e-4),
            per_device_train_batch_size=kwargs.get(
                "per_device_train_batch_size", 4 * 256
            ),
            per_device_eval_batch_size=kwargs.get(
                "per_device_eval_batch_size", 4 * 256
            ),
            num_train_epochs=kwargs.get("num_train_epochs", 3),
            weight_decay=0.01,
            use_cpu=kwargs.get("use_cpu", True),
            data_seed=42,
            seed=42,
            disable_tqdm=False,
            full_determinism=True,
            save_total_limit=11,
            save_safetensors=True,
            do_train=True,
            do_eval=True,
            label_names=[
                "labels",
            ],
        )


@dataclass
class TextEncoderSplit:
    train_dataset: ListDataset
    eval_dataset: ListDataset
    test_dataset: ListDataset = field(init=False)


class TextEncoderTrainer(Trainer):
    roc_auc_score = evaluate.load("roc_auc")

    def __init__(
        self,
        model: TextEncoderModel,
        training_arguments: TextEncoderTrainingArguments,
        dataset_split: TextEncoderSplit,
        **kwargs,
    ):

        super().__init__(
            model=model,
            args=training_arguments,
            data_collator=TextDataCollatorForList(),
            train_dataset=dataset_split.train_dataset,
            eval_dataset=dataset_split.eval_dataset,
            compute_metrics=TextEncoderTrainer.compute_metrics,
        )
        self._save_path = kwargs.get("save_path", "./saved")
        self._kwargs = kwargs

    def save_all_metrics(self, eval_dataset: ListDataset):
        metrics = self.predict(test_dataset=eval_dataset)
        self.save_metrics(split="all", metrics=metrics)

    @staticmethod
    def compute_metrics(eval_pred: EvalPrediction):
        (logits, labels), _ = eval_pred
        prediction_scores = logits * 2 - 1
        return TextEncoderTrainer.roc_auc_score.compute(
            prediction_scores=prediction_scores, references=labels
        )


class TextEncoderTokenizer(CountVectorizer):
    _filename = "/tokenizer.dill"

    def __init__(
        self,
    ):
        super().__init__(stop_words=["на", "для"], dtype=int)
        self._filename = "/tokenizer.dill"

    def save(self, save_path: str = "./saved/"):
        with open(save_path + TextEncoderTokenizer._filename, "wb") as fn:
            dill.dump(self, fn)

    def tokenize(self, text: str) -> list[int]:
        vec = self.transform([text])
        return vec[0].nonzero()[1]

    @staticmethod
    def load(save_path: str = "./saved/"):
        with open(save_path + TextEncoderTokenizer._filename, "rb") as fn:
            return dill.load(fn)


@dataclass
class TextEncoderDatasets:
    """
    Take list of search_texts
    Do negative sampling.
    Do train-eval split.

    """

    search_texts_vecs: csr_matrix

    @property
    def texts_size(self) -> int:
        return self.search_texts_vecs.shape[0]

    @property
    def tokens_size(self) -> int:
        return self.search_texts_vecs.shape[1]

    def neg_choice(
        self,
        token_ids: tuple[int],
        neg_per_sample: int,
    ) -> list[int]:
        """
        --
        Return:
        """
        neg_idx = np.argwhere(
            self.search_texts_vecs[:, token_ids].sum(axis=1).A.ravel() == 0
        )
        distr = np.float32(
            np.log10(self.search_texts_vecs.sum(axis=1).A.ravel()[neg_idx] + 1).ravel()
        )
        distr = distr / distr.sum()
        return tuple(
            np.random.choice(neg_idx.ravel(), neg_per_sample, p=distr).tolist()
        )

    def make_negative_samples(
        self, neg_per_sample: int
    ) -> list[tuple[int, tuple[int], tuple[int]]]:
        """
        pos_text_id, [neg_text_ids], [token_ids]
        """
        with ThreadPoolExecutor() as pool:
            pos_neg_token_text_ids = list(
                pool.map(
                    lambda tu: (
                        tu[0],
                        self.neg_choice(
                            token_ids=tuple(map(int, tu[1].nonzero()[1])),
                            neg_per_sample=neg_per_sample,
                        ),
                        tuple(map(int, tu[1].nonzero()[1])),
                    ),
                    enumerate(self.search_texts_vecs),
                )
            )
        return pos_neg_token_text_ids

    def create_dataset(
        self, pos_neg_token_text_ids: list[tuple[int, tuple[int], tuple[int]]]
    ) -> list[list[int, int, int]]:
        """
        Expand list of pos_neg_token_text_ids to tuple of text_id, [token_ids], label
        """
        dataset = []
        for pos_text_id, neg_text_ids, token_ids in tqdm(pos_neg_token_text_ids):
            dataset.append((pos_text_id, token_ids, 1))
            for neg_text_id in neg_text_ids:
                dataset.append((neg_text_id, token_ids, -1))
        return dataset

    def __train_eval_split(self, dataset: ListDataset) -> TextEncoderSplit:
        generator = torch.Generator().manual_seed(42)
        train_dataset, eval_dataset = torch.utils.data.random_split(
            dataset, [0.95, 0.05], generator=generator
        )
        return TextEncoderSplit(train_dataset, eval_dataset)

    def save(self, save_path: str = "./saved/"):
        with open(save_path + "text_datasets.dill", "wb") as fn:
            dill.dump(self, fn)

    def split(self, neg_per_sample: int = 3) -> TextEncoderSplit:
        """
        Make negative sampling and split dataset
        ---
        Return:
        train_dataset, eval_dataset
        """
        pos_neg_token_text_ids = self.make_negative_samples(
            neg_per_sample=neg_per_sample
        )
        dataset = ListDataset(self.create_dataset(pos_neg_token_text_ids))
        self._datasets = self.__train_eval_split(dataset)
        return self._datasets
