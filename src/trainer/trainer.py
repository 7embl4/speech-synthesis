import torch
from torch import nn

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.g_optimizer.zero_grad()
            self.d_optimizer.zero_grad()

        # discriminator step
        self._unfreeze(self.model.discriminator)
        self._freeze(self.model.generator)

        outputs = self.model.discriminate(**batch)
        batch.update(outputs)

        d_loss = self.d_criterion(**batch)
        batch.update(d_loss)

        if self.is_train:
            batch["d_loss"].backward()
            self._clip_grad_norm()
            self.d_optimizer.step()
            if self.d_lr_scheduler is not None:
                self.d_lr_scheduler.step()

        # generator step
        self._unfreeze(self.model.generator)
        self._freeze(self.model.discriminator)

        outputs = self.model.generate(**batch)
        batch.update(outputs)

        g_loss = self.g_criterion(**batch)
        batch.update(g_loss)

        if self.is_train:
            batch["g_loss"].backward()
            self._clip_grad_norm()
            self.g_optimizer.step()
            if self.g_lr_scheduler is not None:
                self.g_lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_audio(
        self, audio: torch.Tensor, gen_audio: torch.Tensor, log_count=10, **batch
    ):
        """
        Logs `log_count` of audios in batch
        """
        for i in range(log_count):
            real = audio[i].cpu().detach()
            gen = gen_audio[i].cpu().detach()

            self.writer.add_audio(f"real_idx{i}_step{self.writer.step}", real, 22050)
            self.writer.add_audio(
                f"generated_idx{i}_step{self.writer.step}", gen, 22050
            )

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        if mode == "train":
            self._log_audio()
        else:
            self._log_audio()

    def _freeze(self, model: nn.Module):
        for param in model.parameters():
            param.requires_grad_(False)

    def _unfreeze(self, model: nn.Module):
        for param in model.parameters():
            param.requires_grad_(True)
