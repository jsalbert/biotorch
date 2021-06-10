import torch
from biotorch.training.functions import train, test


class Trainer:
    def __init__(self,
                 model,
                 mode,
                 loss_function,
                 optimizer,
                 train_dataloader,
                 test_dataloader,
                 device,
                 epochs,
                 multi_gpu=False,
                 display_iterations=500):

        self.model = model
        self.mode = mode
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        self.epochs = epochs
        self.multi_gpu = multi_gpu
        self.display_iterations = display_iterations

    def run(self):
        for epoch in range(self.epochs):

            train(
                model=self.model,
                mode=self.mode,
                loss_function=self.loss_function,
                optimizer=self.optimizer,
                train_dataloader=self.train_dataloader,
                device=self.device,
                epoch=epoch
            )

            acc, loss = test(
                model=self.model,
                loss_function=self.loss_function,
                test_dataloader=self.test_dataloader,
                device=self.device,
            )

            # Remember best acc@1 and save checkpoint
            if acc > self.best_acc:
                self.best_acc = max(acc, self.best_acc)
                if self.multi_gpu:
                    torch.save(self.model.module, os.path.join(self.output_dir, 'model_best_acc.pth'))
                else:
                    torch.save(self.model, os.path.join(self.output_dir, 'model_best_acc.pth'))
            torch.save(self.model, os.path.join(self.output_dir, 'latest_model.pth'))
