import torch
import os
import csv
import datetime
import time
from contextlib import contextmanager

from utils.dist_utils import dist_print, dist_tqdm, synchronize
from utils.factory import get_metric_dict, get_loss_dict, get_optimizer, get_scheduler
from utils.metrics import update_metrics, reset_metrics
from utils.common import calc_loss, get_model, get_train_loader, inference, merge_config, save_model, cp_projects
from utils.common import get_work_dir, get_logger

@contextmanager
def timer(name):
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        dist_print(f"{name} took {end - start:.2f} seconds")

class CSVLogger:
    """Minimal CSV logger for steps and epochs."""
    def __init__(self, work_dir, rank0=True):
        self.rank0 = rank0
        self.work_dir = work_dir
        if self.rank0:
            os.makedirs(self.work_dir, exist_ok=True)
            self.step_path = os.path.join(self.work_dir, "train_steps.csv")
            self.epoch_path = os.path.join(self.work_dir, "train_epochs.csv")
            if not os.path.exists(self.step_path):
                with open(self.step_path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["global_step","epoch","batch_idx","loss","lr","top1","top2","time"])
            if not os.path.exists(self.epoch_path):
                with open(self.epoch_path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["epoch","avg_loss","epoch_time","lr_end"])

    def log_step(self, global_step, epoch, batch_idx, loss, lr, top1=None, top2=None):
        if not self.rank0: return
        with open(self.step_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([global_step, epoch, batch_idx, float(loss), float(lr),
                        (None if top1 is None else float(top1)),
                        (None if top2 is None else float(top2)),
                        time.time()])

    def log_epoch(self, epoch, avg_loss, epoch_time, lr_end):
        if not self.rank0: return
        with open(self.epoch_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([epoch, float(avg_loss), float(epoch_time), float(lr_end)])

class EarlyStopper:
    """Simple early stopping on a scalar (e.g., train avg_loss or val_loss)."""
    def __init__(self, patience=10, min_delta=1e-3, warmup_epochs=5):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.warmup_epochs = int(warmup_epochs)
        self.best = float('inf')
        self.num_bad = 0

    def step(self, value, epoch):
        """Return True if should stop early."""
        improved = (self.best - value) > self.min_delta
        if improved:
            self.best = value
            self.num_bad = 0
        else:
            if epoch + 1 > self.warmup_epochs:
                self.num_bad += 1
        return self.num_bad >= self.patience

class TrainingSession:
    def __init__(self, cfg, args, distributed):
        self.cfg = cfg
        self.args = args
        self.distributed = distributed
        self.start_time = time.time()

        # ---- Early Stop configs (can be added to your config file) ----
        self.use_es = getattr(cfg, "early_stop", True)
        self.es_patience = getattr(cfg, "es_patience", 10)
        self.es_min_delta = getattr(cfg, "es_min_delta", 1e-3)
        self.es_warmup_epochs = getattr(cfg, "es_warmup_epochs", 5)

        # dataloader + model
        self.train_loader = get_train_loader(cfg)
        self.net = get_model(cfg)

        if distributed:
            self.net = torch.nn.parallel.DistributedDataParallel(
                self.net, device_ids=[args.local_rank]
            )

        self.optimizer = get_optimizer(self.net, cfg)
        self._load_weights()

        self.scheduler = get_scheduler(self.optimizer, cfg, len(self.train_loader))
        self.metric_dict = get_metric_dict(cfg)
        self.loss_dict = get_loss_dict(cfg)

        # rank-0 workspace + logger
        if getattr(args, "local_rank", 0) == 0:
            self.work_dir = get_work_dir(cfg)  # where checkpoints & logs go
            self.logger = get_logger(self.work_dir, cfg)  # existing txt logger
            cp_projects(cfg.auto_backup, self.work_dir)
            self.csv = CSVLogger(self.work_dir, rank0=True)
        else:
            self.work_dir = None
            self.logger = None
            self.csv = CSVLogger(".", rank0=False)

        self.best_loss = float('inf')
        self.resume_epoch = self._get_resume_epoch()
        self.early_stopper = EarlyStopper(self.es_patience, self.es_min_delta, self.es_warmup_epochs) if self.use_es else None

    def _load_weights(self):
        if getattr(self.cfg, "finetune", None):
            dist_print('Finetuning from:', self.cfg.finetune)
            state_all = torch.load(self.cfg.finetune, map_location='cpu')
            state_dict = state_all['model'] if isinstance(state_all, dict) and 'model' in state_all else state_all
            self.net.load_state_dict(state_dict, strict=False)
        if getattr(self.cfg, "resume", None):
            dist_print('Resuming from:', self.cfg.resume)
            resume_dict = torch.load(self.cfg.resume, map_location='cpu')
            self.net.load_state_dict(resume_dict['model'])
            if 'optimizer' in resume_dict:
                self.optimizer.load_state_dict(resume_dict['optimizer'])

    def _get_resume_epoch(self):
        if getattr(self.cfg, "resume", None):
            filename = os.path.basename(self.cfg.resume)
            digits = "".join(ch for ch in filename if ch.isdigit())
            if len(digits) >= 1:
                try:
                    return int(digits) + 1
                except:
                    return 0
        return 0

    def _get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def train_epoch(self, epoch):
        self.net.train()
        progress_bar = dist_tqdm(self.train_loader, desc=f"Epoch {epoch+1:03d}/{self.cfg.epoch}")

        epoch_loss = 0.0
        n_batches = len(self.train_loader)

        for batch_idx, data_label in enumerate(progress_bar):
            global_step = epoch * n_batches + batch_idx

            # forward + loss
            results = inference(self.net, data_label, self.cfg.dataset)
            loss = calc_loss(self.loss_dict, results, self.logger, global_step, epoch)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step(global_step)

            # metric preview mỗi 10 bước
            top1 = top2 = None
            if global_step % 10 == 0:
                reset_metrics(self.metric_dict)
                update_metrics(self.metric_dict, results)
                top1 = self.metric_dict["op"][0].get()
                top2 = self.metric_dict["op"][1].get()
                if hasattr(progress_bar, 'set_postfix'):
                    progress_bar.set_postfix(loss=f'{loss.item():.3f}', top1=f'{top1:.3f}', top2=f'{top2:.3f}')

            # CSV step log (rank0 only)
            lr_now = self._get_lr()
            self.csv.log_step(global_step, epoch, batch_idx, loss.item(), lr_now, top1, top2)

            epoch_loss += loss.item()

        return epoch_loss / max(1, n_batches)

    def _plot_curves(self):
        """Create simple PNG plots from CSV logs (rank0 only)."""
        if getattr(self.args, "local_rank", 0) != 0:
            return
        try:
            import pandas as pd
            import matplotlib.pyplot as plt

            steps_csv = os.path.join(self.work_dir, "train_steps.csv")
            epochs_csv = os.path.join(self.work_dir, "train_epochs.csv")

            # Loss (steps)
            if os.path.exists(steps_csv):
                df = pd.read_csv(steps_csv)
                if "loss" in df.columns and "global_step" in df.columns:
                    plt.figure()
                    plt.plot(df["global_step"], df["loss"])
                    plt.xlabel("global_step"); plt.ylabel("loss"); plt.title("Training Loss (step)")
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.work_dir, "loss_curve_step.png"))
                    plt.close()

                if "lr" in df.columns:
                    plt.figure()
                    plt.plot(df["global_step"], df["lr"])
                    plt.xlabel("global_step"); plt.ylabel("learning_rate"); plt.title("LR Schedule")
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.work_dir, "lr_curve.png"))
                    plt.close()

                if "top1" in df.columns and df["top1"].notna().any():
                    plt.figure()
                    df2 = df.dropna(subset=["top1"])
                    plt.plot(df2["global_step"], df2["top1"])
                    plt.xlabel("global_step"); plt.ylabel("top1"); plt.title("Top-1 (approx)")
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.work_dir, "top1_curve.png"))
                    plt.close()

                    if "top2" in df.columns and df2["top2"].notna().any():
                        plt.figure()
                        plt.plot(df2["global_step"], df2["top2"])
                        plt.xlabel("global_step"); plt.ylabel("top2"); plt.title("Top-2 (approx)")
                        plt.tight_layout()
                        plt.savefig(os.path.join(self.work_dir, "top2_curve.png"))
                        plt.close()

            # Loss (epochs)
            if os.path.exists(epochs_csv):
                df = pd.read_csv(epochs_csv)
                if "avg_loss" in df.columns and "epoch" in df.columns:
                    plt.figure()
                    plt.plot(df["epoch"], df["avg_loss"])
                    plt.xlabel("epoch"); plt.ylabel("avg_loss"); plt.title("Training Loss (epoch)")
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.work_dir, "loss_curve_epoch.png"))
                    plt.close()

        except Exception as e:
            dist_print(f"[warn] plotting failed: {e}")

    def run(self):
        dist_print(f"Starting training at {datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')}")
        dist_print(f"Epochs: {self.cfg.epoch}, Batches per epoch: {len(self.train_loader)}")

        for epoch in range(self.resume_epoch, self.cfg.epoch):
            epoch_start_time = time.time()

            avg_loss = self.train_epoch(epoch)
            self.train_loader.reset()

            # save checkpoint (rank0)
            if getattr(self.args, "local_rank", 0) == 0:
                if avg_loss < self.best_loss:
                    self.best_loss = avg_loss
                    save_model(self.net, self.optimizer, epoch, self.work_dir, self.distributed, filename='best_model.pth')
                if (epoch + 1) % 10 == 0:
                    save_model(self.net, self.optimizer, epoch, self.work_dir, self.distributed, filename=f'epoch_{epoch+1:03d}.pth')

            epoch_time = time.time() - epoch_start_time
            if epoch % 5 == 0 or epoch == self.cfg.epoch - 1:
                dist_print(f"Epoch {epoch+1:03d}/{self.cfg.epoch} - Loss: {avg_loss:.4f} - Time: {epoch_time:.1f}s")

            # CSV epoch log (rank0)
            if getattr(self.args, "local_rank", 0) == 0:
                self.csv.log_epoch(epoch, avg_loss, epoch_time, self._get_lr())

            # ---- Early stopping check ----
            if self.early_stopper is not None:
                if self.early_stopper.step(avg_loss, epoch):
                    dist_print(f"[EarlyStop] No improvement > {self.es_min_delta} for {self.es_patience} epochs after warmup={self.es_warmup_epochs}. Stopping at epoch {epoch+1}.")
                    break

        total_time = time.time() - self.start_time
        dist_print(f"\nTraining completed in {datetime.timedelta(seconds=int(total_time))}")
        dist_print(f"Best loss achieved: {self.best_loss:.4f}")

        # auto-plot curves to PNGs in work_dir
        self._plot_curves()

        if self.logger is not None:
            self.logger.close()

def main():
    torch.backends.cudnn.benchmark = True
    args, cfg = merge_config()

    distributed = False
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    training_session = TrainingSession(cfg, args, distributed)
    training_session.run()

if __name__ == "__main__":
    main()
