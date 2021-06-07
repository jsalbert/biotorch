import time
import torch
from biotorch.training.metrics import accuracy, ProgressMeter, AverageMeter


class Trainer:
    def __init__(self,
                 model,
                 mode,
                 loss_function,
                 optimizer,
                 train_dataloader,
                 device,
                 epochs,
                 display_iterations=500):

        self.model = model
        self.mode = mode
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.device = device
        self.epochs = epochs
        self.display_iterations = display_iterations

    def run(self):
        for epoch in range(self.epochs):

            train(
                model=self.model,
                mode=self.mode,
                loss_function=self.loss_function,
                optimizer=self.optimizer,
                train_dataloader=self.train_dataloader,
                device=device,
                epoch=epoch
            )

            acc, loss = test(
                model=self.model,
                loss_function=self.loss_function,
                test_dataloader=self.test_dataloader,
                device=device,
            )

            # Remember best acc@1 and save checkpoint
            if acc > self.best_acc:
                self.best_acc = max(acc, self.best_acc)
                # if self.multi_gpu:
                # torch.save(self.model.module, os.path.join(self.output_dir, 'model_best_acc.pth'))
                # else:
                # torch.save(self.model, os.path.join(self.output_dir, 'model_best_acc.pth'))
            # torch.save(self.model, os.path.join(self.output_dir, 'latest_model.pth'))

