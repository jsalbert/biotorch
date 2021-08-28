import torch
import torch.nn as nn
import numpy as np


from tqdm.auto import tqdm, trange


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))

    def forward(self, x):
        # Broadcasting
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (x - mean) / std


def add_data_normalization(model, mean, std):
    # We can't use torch.transforms because it supports only non-batch images.
    norm_layer = Normalize(mean=mean, std=std)

    model_ = torch.nn.Sequential(
        norm_layer,
        model
    )
    return model_


def apply_attack_on_dataset(model, dataloader, attack, epsilons, device, verbose=True):
    robust_accuracy = []
    c_a = []
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, pre = torch.max(outputs.data, 1)
        correct_predictions = pre == labels
        c_a.append((correct_predictions.sum() / len(correct_predictions)).cpu().numpy())

    clean_accuracy = np.mean(c_a)
    print('Clean accuracy: ', clean_accuracy)

    for epsilon in epsilons:
        attack.eps = epsilon
        r_a = []
        if verbose:
            print("Epsilon: ", epsilon)
            t = trange(len(dataloader))
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            adv_images = attack(images, labels)
            outputs = model(adv_images)
            _, pre = torch.max(outputs.data, 1)
            correct_predictions = pre == labels
            r_a.append((correct_predictions.sum() / len(correct_predictions)).cpu().numpy())
            if verbose:
                t.update(1)

        robust_acc = np.mean(r_a)
        if verbose:
            print('Robust accuracy: ', robust_acc)
        robust_accuracy.append(robust_acc)

    return clean_accuracy, robust_accuracy


def apply_attack_on_batch(model, images, labels, attack, device):
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, pre = torch.max(outputs.data, 1)
    correct_predictions = pre == labels
    correct_predictions = correct_predictions.cpu().numpy()
    clean_accuracy = (correct_predictions.sum() / len(correct_predictions))

    adv_images = attack(images, labels)
    outputs = model(adv_images)
    _, pre = torch.max(outputs.data, 1)
    correct_predictions_adv = pre == labels
    correct_predictions_adv = correct_predictions_adv.cpu().numpy()
    robust_accuracy = (correct_predictions_adv.sum() / len(correct_predictions_adv))
    adversarial_success = []

    for pred_c, pred_r in zip(correct_predictions, correct_predictions_adv):
        if pred_c and not pred_r:
            adversarial_success.append(True)
        else:
            adversarial_success.append(False)

    print('Clean Accuracy on Batch: {}%'.format(clean_accuracy))
    print('Robust Accuracy on Batch: {}%'.format(robust_accuracy))
    return adv_images.cpu(), adversarial_success, clean_accuracy, robust_accuracy
