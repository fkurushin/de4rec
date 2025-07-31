from dataclasses import dataclass, field
from typing import Iterable, Optional

import evaluate
import numpy as np
import torch
import scipy.sparse
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


class DataCollatorForList:
    """
    Convert DataList to named tensors and move to device
    """

    def __call__(self, batch):
        tbatch = torch.tensor(batch)
        return {
            "user_ids": tbatch[:, 0:1],
            "item_ids": tbatch[:, 1:2],
            "labels": tbatch[:, 2],
        }


@dataclass
class DualEncoderOutput(ModelOutput):
    """
    Output of the DualEncoder model
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    labels: Optional[torch.LongTensor] = None


@dataclass
class DualEncoderSplit:
    train_dataset: ListDataset
    eval_dataset: ListDataset
    test_dataset: ListDataset = field(init=False)


class DualEncoderConfig(PretrainedConfig):
    model_type = "DualEncoder"

    def __init__(
        self,
        users_size: int = None,
        items_size: int = None,
        embedding_dim: int = None,
        margin: float = 0.85,
        max_norm: float = 5.0,
        **kwargs,
    ):
        self.users_size = users_size
        self.items_size = items_size
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.max_norm = max_norm
        super().__init__(**kwargs)


class DualEncoderModel(PreTrainedModel):
    config_class = DualEncoderConfig

    def __init__(self, config: DualEncoderConfig):
        super().__init__(config)
        self.user_embeddings = torch.nn.Embedding(
            config.users_size, config.embedding_dim, max_norm=config.max_norm
        )
        self.item_embeddings = torch.nn.Embedding(
            config.items_size, config.embedding_dim, max_norm=config.max_norm
        )
        self.cel = torch.nn.CosineEmbeddingLoss(margin=config.margin, reduction="mean")
        self.cs = torch.nn.CosineSimilarity()

    def forward(self, **kwargs) -> DualEncoderOutput:
        user_embs = self.user_embeddings(kwargs["user_ids"]).squeeze(1)
        item_embs = self.item_embeddings(kwargs["item_ids"]).squeeze(1)
        logits = self.cs(user_embs, item_embs)
        loss = self.cel(user_embs, item_embs, kwargs["labels"])
        return DualEncoderOutput(loss=loss, logits=logits, labels=kwargs["labels"])

    def recommend_topk_by_user_ids(
        self, user_ids: list[int], top_k: int
    ) -> list[list[int]]:
        """
        Recommend by known user_id. It have to be in the model.
        You may give a list of known user_ids
        ---
        Return:
        A list of list of item ids
        """
        recommended_item_ids = []
        for user_id in user_ids:
            recommended_item_ids.append(
                torch.topk(
                    self.cs(
                        self.user_embeddings.weight[user_id, :],
                        self.item_embeddings.weight,
                    ).detach(),
                    k=top_k,
                ).indices.tolist()
            )
        return recommended_item_ids

    def recommend_topk_by_item_ids(self, item_ids: list[int], top_k: int) -> list[int]:
        """
        In case of a new user (which has no embedding) may take a list of item_ids of the new user, average them, and use as an surrogate new user embedding.
        ---
        Return:
        A list of item_ids
        """
        return torch.topk(
            self.cs(
                self.item_embeddings.weight[item_ids, :].mean(dim=0),
                self.item_embeddings.weight,
            ).detach(),
            k=top_k,
        ).indices.tolist()


class DualEncoderRecommender:
    def __init__(self, model: DualEncoderModel):
        model.eval()
        with torch.no_grad():
            self.user_normed_embs = (
                model.user_embeddings.weight
                / torch.functional.norm(model.user_embeddings.weight, dim=1).unsqueeze(
                    -1
                )
            )
            self.item_normed_embs = (
                model.item_embeddings.weight
                / torch.functional.norm(model.item_embeddings.weight, dim=1).unsqueeze(
                    -1
                )
            )

    def batch_recommend_topk_by_user_ids(
        self, user_ids: list[int], top_k: int, batch_size: int
    ) -> list[list[int]]:
        recommended_item_ids = []
        user_ids_size = len(user_ids)
        for batch in tqdm(range(0, user_ids_size, batch_size)):
            recommended_item_ids += (
                torch.matmul(
                    self.user_normed_embs[user_ids[batch : batch + batch_size]],
                    self.item_normed_embs.T,
                )
                .topk(top_k, dim=1)
                .indices.detach()
                .tolist()
            )

        return recommended_item_ids

    def batch_recommend_topk_by_item_ids(
        self, item_ids_list: list[list[int]], top_k: int, batch_size: int
    ) -> list[list[int]]:
        recommended_item_ids = []
        item_ids_list_size = len(item_ids_list)
        for batch in tqdm(range(0, item_ids_list_size, batch_size)):
            batch_item_id_meaned = []
            for item_ids in item_ids_list[batch : batch + batch_size]:
                batch_item_id_meaned.append(self.item_normed_embs[item_ids].mean(dim=0))

            recommended_item_ids += (
                torch.matmul(
                    torch.stack(batch_item_id_meaned),
                    self.item_normed_embs.T,
                )
                .topk(top_k, dim=1)
                .indices.detach()
                .tolist()
            )

        return recommended_item_ids


class DualEncoderTrainingArguments(TrainingArguments):
    def __init__(self, **kwargs):
        self._kwargs = kwargs
        super().__init__(
            output_dir=kwargs.get("output_dir", "./results"),
            eval_strategy=kwargs.get("eval_strategy", "steps"),
            logging_steps=kwargs.get("logging_steps", 10000),
            prediction_loss_only=False,
            save_strategy="best",
            load_best_model_at_end=True,
            metric_for_best_model=kwargs.get("metric_for_best_model", "eval_loss"),
            learning_rate=kwargs.get("learning_rate", 2e-3),
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


class DualEncoderTrainer(Trainer):
    roc_auc_score = evaluate.load("roc_auc")

    def __init__(
        self,
        model: DualEncoderModel,
        training_arguments: DualEncoderTrainingArguments,
        dataset_split: DualEncoderSplit,
        **kwargs,
    ):

        super().__init__(
            model=model,
            args=training_arguments,
            data_collator=DataCollatorForList(),
            train_dataset=dataset_split.train_dataset,
            eval_dataset=dataset_split.eval_dataset,
            compute_metrics=DualEncoderTrainer.compute_metrics,
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
        return DualEncoderTrainer.roc_auc_score.compute(
            prediction_scores=prediction_scores, references=labels
        )


class DualEncoderDatasets:
    """
    Take interactions list : (user_id, item_id) and two dimensions: users_size and items_size.
    Do negative sampling.
    Do train-eval split.

    """

    def __init__(
        self,
        users_size: int,
        items_size: int,
        interactions: list[tuple[int, int]],
        freq_margin: float = 0.01,
        neg_per_sample: int = 2,
        sparse_neg_sampling: bool = False,
    ):

        self._users_size = users_size
        self._items_size = items_size
        self._freq_margin = freq_margin
        self._neg_per_sample = neg_per_sample

        self.__approximate_dataset_size = len(interactions) * (neg_per_sample + 1)

        if sparse_neg_sampling:
            interactions_sparse, user_indices = self.__list_to_sparse(interactions)
            self.__dataset_split: DualEncoderSplit = self.__train_eval_split(
            self.__add_negative_samples_sparse(interactions_sparse, user_indices)
        )
        else:
            self.__dataset_split: DualEncoderSplit = self.__train_eval_split(
            self.__add_negative_samples(interactions)
        )
        

    @property
    def dataset_split(
        self,
    ) -> DualEncoderSplit:
        return self.__dataset_split

    @property
    def users_size(
        self,
    ) -> int:
        return self._users_size

    @property
    def items_size(
        self,
    ) -> int:
        return self._items_size

    @property
    def freq_margin(
        self,
    ) -> int:
        return self._freq_margin

    @property
    def neg_per_sample(
        self,
    ) -> int:
        return self._neg_per_sample
    

    def __list_to_sparse(
        self, interactions: list[tuple[int, int]]
    ) -> scipy.sparse.csr_matrix:
        """
        Convert list of interactions to sparse matrix.
        Maps original user/item IDs to sequential indices for embedding table.

        We can make the csr_matrix if data was gropued by user ids (i.e. indptr)

        Return:
            scipy.sparse.coo_matrix
        """
        if not interactions:
            return scipy.sparse.coo_matrix((0, 0))
            
        interactions = np.array(interactions)
        user_indices = interactions[:, 0]
        item_indices = interactions[:, 1]
        data = np.ones(len(interactions))
        coo_martrix = scipy.sparse.coo_matrix(
            (data, (user_indices, item_indices)), 
            shape=(self._users_size, self._items_size)
        )
        return coo_martrix.tocsr(), user_indices

    def __make_pos_distributions(
        self, interactions: list[tuple[int, int]]
    ) -> tuple[np.array, dict]:
        freq_dist = np.ones(self.items_size)
        pos_interactions = {}
        for user_id, item_id in interactions:
            freq_dist[item_id] = freq_dist[item_id] + 1
            pos_interactions[user_id] = pos_interactions.get(user_id, []) + [
                item_id,
            ]

        return freq_dist, pos_interactions
    
    def __make_freq_dist(self, interactions: scipy.sparse.csr_matrix) -> np.array:
        return interactions.sum(axis=0).A.ravel()

    def __add_negative_samples_sparse(
        self, interactions: scipy.sparse.csr_matrix, user_indices: np.array
    ) -> list[tuple[int, int, int]]:
        """
        Efficient vectorized negative sampling.
        """
        # 1. Calculate number of negative samples needed per user
        pos_per_user = interactions.sum(axis=1).A.ravel()
        neg_per_user = pos_per_user * self.neg_per_sample
        total_neg_samples = int(neg_per_user.sum())
        
        # 2. Get frequency distribution for negative sampling
        freq_dist = self.__make_freq_dist(interactions)
        freq_margin_num = int(len(freq_dist) * self.freq_margin)
        negative_item_ids = np.argsort(freq_dist)[-freq_margin_num:]
        
        neg_freq_dist = freq_dist[negative_item_ids]
        neg_freq_dist = neg_freq_dist / neg_freq_dist.sum()
        
        # 3. Generate random negative item IDs
        sampled_neg_items = np.random.choice(
            negative_item_ids,
            size=total_neg_samples,
            p=neg_freq_dist
        )
        
        # 4. Create user indices for negative samples
        user_indices = np.repeat(user_indices, self.neg_per_sample)
        
        # 5. Create negative samples sparse matrix
        negative_interactions = scipy.sparse.coo_matrix(
            (np.ones(len(user_indices)), (user_indices, sampled_neg_items)),
            shape=(self.users_size, self.items_size)
        ).tocsr()
        
        # 6. Subtract positive interactions from negative (remove collisions)
        negative_interactions = negative_interactions - interactions
        negative_interactions.data = np.maximum(negative_interactions.data, 0)
        
        # 7. Convert to final format using sparse matrix attributes
        # Positive samples: label = 1
        pos_users, pos_items = interactions.nonzero()
        pos_labels = np.ones(len(pos_users), dtype=int)
        
        # Negative samples: label = -1
        neg_users, neg_items = negative_interactions.nonzero()
        neg_labels = -np.ones(len(neg_users), dtype=int)
        
        # Combine all samples
        all_users = np.concatenate([pos_users, neg_users])
        all_items = np.concatenate([pos_items, neg_items])
        all_labels = np.concatenate([pos_labels, neg_labels])
        
        # Convert to list of tuples in one go
        return list(zip(all_users, all_items, all_labels))

    def __add_negative_samples(
        self, interactions: list[tuple[int, int]]
    ) -> Iterable[tuple[int, int, int]]:

        _freq_dist, pos_interactions = self.__make_pos_distributions(interactions)

        freq_margin_num = int(len(_freq_dist) * self.freq_margin)
        negative_item_ids = np.argsort(_freq_dist)[
            -freq_margin_num:
        ]  # time consuming operation, do it once

        for user_id, pos_item_ids in tqdm(pos_interactions.items()):
            user_negative_item_ids = np.setdiff1d(
                negative_item_ids, np.array(pos_item_ids)
            )

            user_negative_freq_dist = np.log10(_freq_dist[user_negative_item_ids])
            user_negative_freq_dist /= user_negative_freq_dist.sum()

            #Cannot take a larger sample than population when 'replace=False'
            n_samples = min(self.neg_per_sample * len(set(pos_item_ids)), len(user_negative_item_ids))
            neg_item_ids = np.random.choice(
                user_negative_item_ids,
                size=n_samples,
                replace=False,
                p=user_negative_freq_dist,
            )
            for item_id in pos_item_ids:
                yield (user_id, item_id, 1)
            for item_id in neg_item_ids:
                yield (user_id, item_id, -1)

    def __train_eval_split(
        self, dataset: Iterable[tuple[int, int, int]]
    ) -> DualEncoderSplit:

        eval_dataset_size = self.__approximate_dataset_size // 20.0  # 5%
        eval_dataset, train_dataset = [], []
        for idx, tup in enumerate(dataset):
            if idx < eval_dataset_size:
                eval_dataset.append(tup)
            else:
                train_dataset.append(tup)

        return DualEncoderSplit(ListDataset(train_dataset), ListDataset(eval_dataset))


if __name__ == "__main__":

    pass
