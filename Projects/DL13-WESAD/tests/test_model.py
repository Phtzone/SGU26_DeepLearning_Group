import torch
from src.models.cnn import CNN1D
from src.training.optimizer import class_weights, make_optimizer
from src.utils.reproducibility import seed_everything


def test_model_shape_backward_optimizer():
    seed_everything(42)
    model = CNN1D()
    assert sum(p.numel() for p in model.parameters()) == 45059
    x = torch.randn(4, 5, 960)
    output = model(x)
    assert output.shape == (4, 3)
    initial = model.network[0].weight.detach().clone()
    optimizer = make_optimizer(model, {'learning_rate': .001, 'weight_decay': .0001})
    torch.nn.CrossEntropyLoss(weight=class_weights([0, 1, 1, 2]))(output, torch.tensor([0, 1, 1, 2])).backward()
    optimizer.step()
    assert not torch.equal(initial, model.network[0].weight)
    assert CNN1D(2)(torch.randn(2, 2, 960)).shape == (2, 3)
