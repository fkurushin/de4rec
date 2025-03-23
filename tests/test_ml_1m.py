import sys

sys.path.append("src")

from de4rec import DualEncoderPipeline
import pytest
from collections.abc import Iterable


@pytest.fixture
def pipeline():
    pipeline = DualEncoderPipeline()
    return pipeline

def test_pipeline(pipeline):
    assert pipeline.run_default()
