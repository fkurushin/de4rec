import sys

sys.path.append("src")
import os

import dill
import pytest
import torch

from de4rec import (
    DualEncoderConfig,
    DualEncoderDatasets,
    DualEncoderModel,
    DualEncoderRecommender,
    DualEncoderSplit,
    DualEncoderTrainer,
    DualEncoderTrainingArguments,
    s3_tools,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("S3_ACCESS_KEY") is None,
    reason="no s3 env found. Set S3_ACCESS_KEY, S3_SECRET, S3_URL S3_BUCKET_NAME S3_MODEL_KEY S3_DATA_KEY",
)


@pytest.fixture
def save_path():
    return "./DualEncoder/"


class TestS3Train:

    @pytest.fixture
    def s3(
        self,
    ):
        return s3_tools()

    def test_s3(self, s3):
        assert s3.s3_client

    def test_s3_bucket(self):
        assert os.environ.get("S3_BUCKET_NAME")
        assert os.environ.get("S3_MODEL_KEY")

    @pytest.fixture
    def bucket_name(self):
        return os.environ.get("S3_BUCKET_NAME")

    @pytest.fixture
    def model_key(self):
        return os.environ.get("S3_MODEL_KEY")

    @pytest.fixture
    def data_key(self):
        return os.environ.get("S3_DATA_KEY")

    @pytest.fixture
    def data_dict(self, s3, bucket_name: str, data_key: str) -> dict:
        data_dict = s3.get_dill_object(bucket_name, data_key)
        assert isinstance(data_dict, dict)
        return data_dict

    @pytest.fixture
    def users (self, data_dict : dict) -> list :
        return data_dict.get("web_user_ids")

    @pytest.fixture
    def datasets(self, data_dict: dict) -> DualEncoderDatasets:
        
        users_size = len(data_dict.get("web_user_ids")) 
        items_size = len(data_dict.get("search_texts")) 
        interactions = []
        for user_id, item_ids in enumerate(data_dict.get("train_interactions")):
            for item_id in item_ids:
                interactions.append((user_id, item_id))

        return DualEncoderDatasets(interactions=interactions,  users_size=users_size, items_size=items_size)

    def test_datasets(self, datasets: DualEncoderDatasets):
        assert datasets.dataset_split.train_dataset.distinct_size() > 100
        assert datasets.dataset_split.eval_dataset.distinct_size() > 100

    @pytest.fixture
    def model(self, datasets: DualEncoderDatasets) -> DualEncoderModel:
        config = DualEncoderConfig(
            users_size=datasets.users_size,
            items_size=datasets.items_size,
            embedding_dim=32,
        )
        return DualEncoderModel(config)

    def test_model(self, model):
        assert sum(p.numel() for p in model.parameters() if p.requires_grad) > 100_000

    def test_trainer(
        self, datasets: DualEncoderDatasets, model: DualEncoderModel, save_path
    ):
        training_arguments = DualEncoderTrainingArguments(
            logging_steps=10000,
            learning_rate=5e-3,
            use_cpu=not torch.cuda.is_available(),
            per_device_train_batch_size=4 * 16 * 256,
            num_train_epochs=12,
        )

        trainer = DualEncoderTrainer(
            model=model,
            training_arguments=training_arguments,
            dataset_split=datasets.dataset_split,
        )
        trainer.train()
        trainer.save_model(save_path)

        assert trainer

    def test_recom_sample(self, model: DualEncoderModel, datasets: DualEncoderDatasets, users : list):
        sample_num = 100
        model.eval()
        with torch.no_grad():
            inference = model.recommend_topk_by_user_ids(
                user_ids= list(range(sample_num)),
                top_k=5,
            )

        assert len(inference) == sample_num

    @pytest.fixture
    def saved_model(self, save_path) -> DualEncoderModel:
        return DualEncoderModel.from_pretrained(save_path)

    def test_recom_batch(
        self, saved_model: DualEncoderModel, datasets: DualEncoderDatasets, save_path
    ):

        assert (
            sum(p.numel() for p in saved_model.parameters() if p.requires_grad) > 1000
        )

        recommender = DualEncoderRecommender(model=saved_model)
        inference = recommender.batch_recommend_topk_by_user_ids(
            user_ids= list(range(datasets.users_size)), top_k=5, batch_size=10_000
        )

        assert len(inference) == datasets.users_size

        dill.dump(inference, open(save_path + "/inference.dill", "wb"))
        dill.dump(datasets, open(save_path + "/datasets.dill", "wb"))

    def test_upload(self, s3, bucket_name: str, model_key: str, save_path: str):
        assert s3.safe_upload_folder(save_path, bucket_name, model_key)


class TestInference:

    @pytest.fixture
    def saved_model(self, save_path) -> DualEncoderModel:
        return DualEncoderModel.from_pretrained(save_path)

    def test_recom_for_unknown_users_batch(
        self,
        saved_model: DualEncoderModel,
    ):

        recommender = DualEncoderRecommender(model=saved_model)
        inference = recommender.batch_recommend_topk_by_item_ids(
            [
                [1, 2, 3],
                [4, 5, 6, 7],
                [
                    22,
                ],
                [
                    33,
                    44,
                ],
            ],
            top_k=5,
            batch_size=2,
        )

        assert len(inference) == 4        
