import sys

sys.path.append("src")

import pytest

from de4rec import (
    DualEncoderModel,
    DualEncoderRecommender,
)


@pytest.fixture
def save_path():
    return "./DualEncoder/"


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


