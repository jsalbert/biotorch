import os
import time
import torch

from biotorch.training.functions import test


class Evaluator:
    def __init__(self,
                 model,
                 mode,
                 loss_function,
                 dataloader,
                 device,
                 output_dir,
                 multi_gpu=False,
                 ):

        self.model = model
        self.mode = mode
        self.output_dir = output_dir
        self.logs_dir = os.path.join(output_dir, 'logs')
        self.loss_function = loss_function
        self.dataloader = dataloader
        self.device = device
        self.multi_gpu = multi_gpu

    def run(self):
        acc, loss = test(
            model=self.model,
            loss_function=self.loss_function,
            test_dataloader=self.dataloader,
            device=self.device,
        )

        return acc.cpu().numpy(), loss
