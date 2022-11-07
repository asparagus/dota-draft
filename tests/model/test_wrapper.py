import torch

from draft.model import mlp
from draft.model import wrapper


def test_symmetry():
    symmetric_config = wrapper.ModelWrapperConfig(
        symmetric=True,
    )
    mlp_config = mlp.MLPConfig(
        num_heroes=20,
        layers=[10, 10],
    )
    symmetric_model = wrapper.ModelWrapper(symmetric_config, module=mlp.MLP(mlp_config))
    draft = torch.tensor([[1,2,3,4,5,6,7,8,9,10]], dtype=torch.long)
    opposite_draft = torch.tensor([[6,7,8,9,10,1,2,3,4,5]], dtype=torch.long)
    symmetric_result = symmetric_model(draft)
    opposite_result = symmetric_model(opposite_draft)
    torch.testing.assert_close(symmetric_result, 1 - opposite_result)
