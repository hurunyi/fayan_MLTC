import os
import torch
from callback.progressbar import ProgressBar
from utils.utils import seed_everything
from torch.nn.utils import clip_grad_norm_


class Trainer(object):
    def __init__(
            self,
            model,
            epochs,
            criterion,
            optimizer,
            lr_scheduler,
            epoch_metrics,
            batch_metrics,
            grad_clip=0.0,
            load_from_checkpoint=False,
            checkpoint_dir=None
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.best = 0
        self.start_epoch = 0
        self.model = model
        self.epochs = epochs
        self.grad_clip = grad_clip
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.epoch_metrics = epoch_metrics
        self.batch_metrics = batch_metrics
        self.model = self.model.to(self.device)
        self.checkpoint_dir = None

        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            self.checkpoint_dir = checkpoint_dir

        if load_from_checkpoint:
            print(f"\nLoading checkpoint: {checkpoint_dir}")
            resume_dict = torch.load(os.path.join(checkpoint_dir, "best_checkpoint.pt"))
            self.best = resume_dict['best']
            self.start_epoch = resume_dict['epoch']
            self.model.load_state_dict(resume_dict["model"])
            self.optimizer.load_state_dict(resume_dict["optimizer"])
            self.lr_scheduler.load_state_dict(resume_dict["lr_scheduler"])
            print(f"\nCheckpoint '{checkpoint_dir}' and epoch {self.start_epoch} loaded")

    def epoch_reset(self):
        self.outputs = []
        self.targets = []
        self.result = {}
        for metric in self.epoch_metrics:
            metric.reset()

    def batch_reset(self):
        self.info = {}
        for metric in self.batch_metrics:
            metric.reset()

    def save_info(self, epoch, best):
        model_save = self.model.module if hasattr(self.model, 'module') else self.model
        state = {
            "model": model_save,
            "epoch": epoch,
            "best": best
        }
        return state

    def valid_epoch(self, data):
        pbar = ProgressBar(n_total=len(data))
        self.epoch_reset()
        self.model.eval()
        with torch.no_grad():
            for step, batch in enumerate(data):
                batch = tuple(t.to(self.device) for t in batch)
                guid, input_ids, input_mask, segment_ids, label_ids = batch
                guid, logits = self.model(guid, input_ids, input_mask, segment_ids)
                self.outputs.append(logits.cpu().detach())
                self.targets.append(label_ids.cpu().detach())
                pbar.batch_step(step=step, info={}, bar_type='Evaluating')
            self.outputs = torch.cat(self.outputs, dim=0).cpu().detach()
            self.targets = torch.cat(self.targets, dim=0).cpu().detach()
            loss = self.criterion(target=self.targets, output=self.outputs)
            self.result['valid_loss'] = loss.item()
            print("\n------------- valid result --------------")
            if self.epoch_metrics:
                for metric in self.epoch_metrics:
                    value = metric(logits=self.outputs, target=self.targets)
                    if value:
                        self.result[f'valid_{metric.name()}'] = value
            if 'cuda' in str(self.device):
                torch.cuda.empty_cache()
            return self.result

    def train_epoch(self, data):
        pbar = ProgressBar(n_total=len(data))
        self.epoch_reset()
        self.model.train()
        for step,  batch in enumerate(data):
            self.batch_reset()
            self.optimizer.zero_grad()
            batch = tuple(t.to(self.device) for t in batch)
            guid, input_ids, input_mask, segment_ids, label_ids = batch
            guid, logits = self.model(guid, input_ids, input_mask, segment_ids)
            loss = self.criterion(output=logits, target=label_ids)
            loss.backward()
            clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.lr_scheduler.step()
            if self.batch_metrics:
                for metric in self.batch_metrics:
                    metric(logits=logits, target=label_ids)
                    self.info[metric.name()] = metric.value()
            self.info['loss'] = loss.item()
            pbar.batch_step(step=step, info=self.info, bar_type='Training')
            self.outputs.append(logits.cpu().detach())
            self.targets.append(label_ids.cpu().detach())
        print("\n------------- train result --------------")
        # epoch metric
        self.outputs = torch.cat(self.outputs, dim=0).cpu().detach()
        self.targets = torch.cat(self.targets, dim=0).cpu().detach()
        if self.epoch_metrics:
            for metric in self.epoch_metrics:
                value = metric(logits=self.outputs, target=self.targets)
                if value:
                    self.result[f'{metric.name()}'] = value
        if "cuda" in str(self.device):
            torch.cuda.empty_cache()
        return self.result

    def train(self, train_data, valid_data, seed):
        seed_everything(seed)
        for epoch in range(self.start_epoch+1, self.start_epoch+self.epochs+1):
            print(f"Epoch {epoch}/{self.start_epoch+self.epochs}")
            train_log = self.train_epoch(train_data)
            valid_log = self.valid_epoch(valid_data)
            logs = dict(train_log, **valid_log)
            show_info = f'\nEpoch: {epoch} - ' + "-".join([f' {key}: {value:.4f} ' for key, value in logs.items()])
            print(show_info)

            # save model
            if logs["valid_f1"] > self.best:
                print(f"Valid_f1 improved from {self.best} to {logs['valid_f1']}")
                self.best = logs["valid_f1"]
                if self.checkpoint_dir is not None:
                    torch.save(
                        {
                            "epoch": epoch,
                            "best": self.best,
                            "model": self.model.state_dict(),
                            "optimizer": self.optimizer.state_dict(),
                            "lr_scheduler": self.lr_scheduler.state_dict()
                        },
                        os.path.join(self.checkpoint_dir, "best_checkpoint.pt")
                    )
                print("Saved the model!")
