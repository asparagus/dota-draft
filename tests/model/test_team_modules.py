import torch

from draft.model import team_modules


def test_split():
    splitter = team_modules.TeamSplitter()
    draft = torch.tensor([[1,2,3,4,5,6,7,8,9,10]], dtype=torch.long)
    radiant_draft, dire_draft = splitter(draft)
    expected_radiant_draft = torch.tensor([[1,2,3,4,5]], dtype=torch.long)
    expected_dire_draft = torch.tensor([[6,7,8,9,10]], dtype=torch.long)
    torch.testing.assert_close(radiant_draft, expected_radiant_draft)
    torch.testing.assert_close(dire_draft, expected_dire_draft)


def test_merge():
    merger = team_modules.TeamMerger()
    radiant_embedding = torch.tensor(
        [
            [
                [1,1,0,0,0],
                [0,1,1,0,0],
                [0,0,1,1,0],
                [0,0,0,1,1],
                [0,0,0,0,2],
            ],
        ]
    )
    dire_embedding = torch.tensor(
        [
            [
                [1,0,0,0,0],
                [0,1,0,0,0],
                [0,0,1,0,0],
                [0,0,0,1,0],
                [0,0,0,0,1],
            ],
        ]
    )
    expected_merge = torch.tensor([[0, 1, 1, 1, 2]])
    merge = merger.forward(radiant_embedding, dire_embedding)
    torch.testing.assert_close(merge, expected_merge)


def test_convolution_dimensions():
    config = team_modules.TeamConvolutionConfig(input_dimension=10, layers=[10], activation=True)
    conv = team_modules.TeamConvolution(config=config)
    draft = torch.rand(1, 10, 10)
    out = conv(draft)
    assert out.shape == draft.shape
