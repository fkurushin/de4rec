import sys

sys.path.append("src")

import numpy as np
import pytest

from de4rec import (
    DualEncoderDatasets,
    DualEncoderConfig,
    DualEncoderModel,
    DualEncoderTrainer,
    DualEncoderTrainingArguments,
    DualEncoderRecommender
)
import torch

class DualEncoderLoadData(DualEncoderDatasets):
    def __init__(self, **kwargs):
    
        _interactions_path = kwargs.get(
            "interactions_path", "dataset/ml-1m/ratings.dat"
        )
        assert _interactions_path
        _interactions = self.load_list_of_int_int_from_path(_interactions_path)
    
        _users_size = max([tu[0] for tu in _interactions]) + 1
        _items_size = max([tu[1] for tu in _interactions]) + 1
        
        super().__init__(
            interactions=_interactions, users_size=_users_size, items_size=_items_size, freq_margin=1.0, neg_per_sample=1
        )
    
    def load_list_of_int_int_from_path(
        self, path: str, sep: str = "::"
    ) -> list[tuple[int, int]]:
        with open(path, "r", encoding="utf-8") as fn:
            res = list(
                map(
                    lambda row: (int(row[0]), int(row[1])),
                    map(
                        lambda row: row.strip().split(sep)[:2],
                        fn.read().strip().split("\n"),
                    ),
                )
            )
        return res

class TestML1M:
    @pytest.fixture
    def save_path(self,):
        return "./ml_1m/"


    @pytest.fixture
    def datasets(self,):
        datasets = DualEncoderLoadData(
            interactions_path="dataset/ml-1m/ratings.dat",
            users_path="dataset/ml-1m/users.dat",
            items_path="dataset/ml-1m/movies.dat",
        )
        return datasets


    def test_datasets(self, datasets):
        assert datasets.items_size == 3953
        assert datasets.users_size == 6041
        assert datasets.dataset_split.train_dataset.distinct_size() > 1
        assert datasets.dataset_split.eval_dataset.distinct_size() > 1

    def test_config(self, save_path):
        config = DualEncoderConfig(users_size=101, items_size=102, embedding_dim=32)
        config.save_pretrained(save_path)


    def test_trainer(self, datasets, save_path):
        config = DualEncoderConfig(
            users_size=datasets.users_size,
            items_size=datasets.items_size,
            embedding_dim=32,
        )
        model = DualEncoderModel(config)

        training_arguments = DualEncoderTrainingArguments(
            logging_steps=1000,
            learning_rate=1e-3,
            use_cpu=not torch.cuda.is_available(),
            per_device_train_batch_size=4 * 256,
        )

        trainer = DualEncoderTrainer(
            model=model, training_arguments=training_arguments, dataset_split=datasets.dataset_split
        )
        trainer.train()
        trainer.save_model(save_path)
        assert trainer

    @pytest.fixture
    def model(self, save_path) -> DualEncoderModel: 
        return DualEncoderModel.from_pretrained(save_path)

    def test_recomm(self, model : DualEncoderModel, datasets):
        assert sum(p.numel() for p in model.parameters() if p.requires_grad) > 1000
        sample = 30
        recommender = DualEncoderRecommender(model=model)
        rec_items = recommender.batch_recommend_topk_by_user_ids(user_ids = list(range(sample)), top_k =  3, batch_size = 10)
        assert len(rec_items) == sample
