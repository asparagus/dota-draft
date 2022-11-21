import pytest

import torch

from draft.model import embedding
from draft.model import mlp
from draft.model import model
from draft.model import team_modules


def build_model(symmetric: bool):
    config = model.ModelConfig(
        embedding_config=embedding.EmbeddingConfig(num_heroes=20, embedding_size=20),
        team_convolution_config=team_modules.TeamConvolutionConfig(input_dimension=20, layers=[20]),
        mlp_config=mlp.MlpConfig(input_dimension=20, layers=[10, 5]),
        symmetric=symmetric,
    )
    return model.Model(config)


def test_symmetry():
    symmetric_model = build_model(symmetric=True)
    draft = torch.tensor([[1,2,3,4,5,6,7,8,9,10]], dtype=torch.long)
    opposite_draft = torch.tensor([[6,7,8,9,10,1,2,3,4,5]], dtype=torch.long)
    symmetric_result = symmetric_model(draft)
    opposite_result = symmetric_model(opposite_draft)
    torch.testing.assert_close(symmetric_result, 1 - opposite_result)


def test_export(tmp_path_factory):
    models = [build_model(symmetric=True), build_model(symmetric=False)]
    path = tmp_path_factory.mktemp('export') / 'model.onnx'
    for model in models:
        model.export(path)
