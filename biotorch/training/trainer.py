import os
import time
import torch


from torch.utils.tensorboard import SummaryWriter
from biotorch.training.functions import train, test
from biotorch.training.metrics import compute_angles_module


class Trainer:
    def __init__(self,
                 model,
                 mode,
                 loss_function,
                 optimizer,
                 lr_scheduler,
                 train_dataloader,
                 val_dataloader,
                 device,
                 epochs,
                 output_dir,
                 multi_gpu=False,
                 display_iterations=500):

        self.model = model
        self.mode = mode
        self.output_dir = output_dir
        self.logs_dir = os.path.join(output_dir, 'logs')
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.epochs = epochs
        self.multi_gpu = multi_gpu
        self.display_iterations = display_iterations
        self.writer = SummaryWriter(self.logs_dir)

    def run(self):
        self.best_acc = 0.0
        for epoch in range(self.epochs):
            t = time.time()
            acc, loss = train(
                model=self.model,
                mode=self.mode,
                loss_function=self.loss_function,
                optimizer=self.optimizer,
                train_dataloader=self.train_dataloader,
                device=self.device,
                epoch=epoch
            )
            self.writer.add_scalar('accuracy/train', acc, epoch)
            self.writer.add_scalar('loss/train', loss, epoch)

            acc, loss = test(
                model=self.model,
                loss_function=self.loss_function,
                test_dataloader=self.val_dataloader,
                device=self.device,
            )
            self.writer.add_scalar('accuracy/test', acc, epoch)
            self.writer.add_scalar('loss/test', loss, epoch)

            # Remember best acc@1 and save checkpoint
            if acc > self.best_acc:
                self.best_acc = max(acc, self.best_acc)
                if self.multi_gpu:
                    torch.save(self.model, os.path.join(self.output_dir, 'model_best_acc.pth'))
                else:
                    torch.save(self.model, os.path.join(self.output_dir, 'model_best_acc.pth'))
            torch.save(self.model, os.path.join(self.output_dir, 'latest_model.pth'))

            if self.mode == 'weight_mirroring':
                layers_alignment = compute_angles_module(self.model)
                self.writer.add_scalars('layer_alignment/train', layers_alignment, epoch)

            total_time = time.time() - t

            # Update scheduler after training epoch
            self.lr_scheduler.step()
            self.writer.add_scalar('time/train', total_time, epoch)
