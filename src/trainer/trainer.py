import torch
from torch import nn

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

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

        Eto moi commenty (net, Boris, ya ne generil ih neironkoi 😤),
            ya ih pisal shtoby samomu ponyatno bylo che delau
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]

        # generator forward
        g_outputs = self.model.generate(**batch)
        batch.update(g_outputs)

        # 1st discriminator forward
        # to get score of generated audio
        d_outputs = self.model.discriminate(
            audio=batch["audio"],
            gen_audio=batch["gen_audio"].detach(),  # detach from generator comp graph
        )
        batch.update(d_outputs)

        if self.is_train:
            # zeroing grads from previous batch
            # and update discriminator weights
            self.d_optimizer.zero_grad()
            d_loss = self.d_criterion(**batch)
            batch.update(d_loss)
            batch["d_loss"].backward()
            self.d_optimizer.step()

            # 2nd discriminator forward without detach
            # to update generator weights
            d_outputs = self.model.discriminate(
                audio=batch["audio"], gen_audio=batch["gen_audio"]
            )
            batch.update(d_outputs)

            # calc generator loss and update it's weights;
            # discriminator grads updated as well,
            # but we don't care, since it will be zeroed in next batch
            self.g_optimizer.zero_grad()
            g_loss = self.g_criterion(**batch)
            batch.update(g_loss)
            batch["g_loss"].backward()
            self.g_optimizer.step()

            # clip norm and decay lr
            self._clip_grad_norm()
            if self.g_lr_scheduler is not None:
                self.g_lr_scheduler.step()
            if self.d_lr_scheduler is not None:
                self.d_lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            if self.is_train:
                metrics.update(loss_name, batch[loss_name].item())
            else:
                metrics.update(loss_name, 0)  # no loss on val

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_audio(
        self, audio: torch.Tensor, gen_audio: torch.Tensor, log_count=10, **batch
    ):
        """
        Logs `log_count` of audios in batch
        """
        N = min(audio.shape[0], log_count)
        for i in range(N):
            real = audio[i].detach().cpu()
            gen = gen_audio[i].detach().cpu()

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
            self._log_audio(**batch)
        else:
            self._log_audio(**batch)

    def _freeze(self, model: nn.Module):
        for param in model.parameters():
            param.requires_grad_(False)

    def _unfreeze(self, model: nn.Module):
        for param in model.parameters():
            param.requires_grad_(True)
