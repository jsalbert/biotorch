import torch
import biotorch.models as models


def check_model(model, input_size):
    model_ = model()
    out = model_(torch.rand(input_size))


def test_fa_models(model_architectures):
    for arch, input_size in model_architectures:
        check_model(models.fa.__dict__[arch], input_size)


def test_dfa_models(model_architectures):
    for arch, input_size in model_architectures:
        check_model(models.dfa.__dict__[arch], input_size)


def test_usf_models(model_architectures):
    for arch, input_size in model_architectures:
        check_model(models.fa.__dict__[arch], input_size)


def test_brsf_models(model_architectures):
    for arch, input_size in model_architectures:
        check_model(models.fa.__dict__[arch], input_size)


def test_frsf_models(model_architectures):
    for arch, input_size in model_architectures:
        check_model(models.fa.__dict__[arch], input_size)
