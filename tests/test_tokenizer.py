import sys

sys.path.append("src")

import pytest

from de4rec import TextEncoderTokenizer


class TestTokenizer:

    @pytest.fixture
    def text(self) -> str:
        return "male t-short"

    @pytest.fixture
    def tokenizer(self) -> TextEncoderTokenizer:
        return TextEncoderTokenizer()

    def test_tokenizer(
        self, tokenizer, text
    ):
        vec  = tokenizer.fit_transform([text])
        assert vec.shape == (1,2)


