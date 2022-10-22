import numpy as np
import torch

from draft.model import simplemodel


def test_symmetry():
    symmetric_config = simplemodel.SimpleModelConfig(
        num_heroes=25,
        dimensions=(1,),
        symmetric=True,
    )
    symmetric_model = simplemodel.SimpleModel(symmetric_config)
    draft = torch.tensor([[1,2,3,4,5,6,7,8,9,10]], dtype=torch.long)
    opposite_draft = torch.tensor([[6,7,8,9,10,1,2,3,4,5]], dtype=torch.long)
    symmetric_result = symmetric_model(draft)
    opposite_result = symmetric_model(opposite_draft)
    torch.testing.assert_close(symmetric_result, 1 - opposite_result)
