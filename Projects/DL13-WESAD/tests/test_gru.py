import pytest
import torch
from src.models import build_model
from src.models.cnn import CNN1D
from src.models.gru import GRUClassifier
from src.utils.config import load_config, experiment_folder
from src.utils.reproducibility import seed_everything


def test_gru_layout_batch_isolation_and_learning():
    seed_everything(42)
    model = GRUClassifier()
    x = torch.randn(3, 5, 960)
    model.eval()
    with torch.no_grad():
        actual = model(x)
        _, hidden = model.gru(x.transpose(1, 2).contiguous())
        assert torch.equal(actual, model.classifier(hidden[-1]))
        torch.testing.assert_close(actual, torch.cat([model(row[None]) for row in x]))
    assert actual.shape == (3, 3) and torch.isfinite(actual).all()
    before = model.gru.weight_ih_l0.detach().clone()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    torch.nn.CrossEntropyLoss()(model(x), torch.tensor([0, 1, 2])).backward()
    assert model.gru.weight_ih_l0.grad is not None
    optimizer.step()
    assert not torch.equal(before, model.gru.weight_ih_l0)
    assert GRUClassifier(input_channels=2)(torch.randn(1, 2, 32)).shape == (1, 3)


def test_factory_preserves_cnn_weights_and_e2_namespace():
    config = load_config('experiments/e1_cnn/config.yaml')
    seed_everything(42)
    previous = CNN1D().state_dict()
    seed_everything(42)
    current = build_model(config, 5).state_dict()
    assert previous.keys() == current.keys()
    assert all(torch.equal(previous[key], current[key]) for key in previous)
    gru_config = load_config('experiments/e2_gru/config.yaml')
    assert isinstance(build_model(gru_config, 5), GRUClassifier)
    assert experiment_folder(gru_config) == 'experiment_e2'
    for section in ('data', 'split', 'training', 'selection', 'runtime'):
        assert gru_config[section] == config[section]


@pytest.mark.parametrize('options', [{'hidden_size': 0}, {'hidden_size': True}, {'dropout': 1}])
def test_invalid_gru_options(options):
    with pytest.raises(ValueError):
        GRUClassifier(**options)
