import torch
import os
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
    """Context manager for timing code blocks"""
    start = time.time()
    yield
    end = time.time()
    dist_print(f"{name} took {end - start:.2f} seconds")

class TrainingSession:
    def __init__(self, cfg, args, distributed):
        self.cfg = cfg
        self.args = args
        self.distributed = distributed
        self.start_time = time.time()
        
        # Initialize components
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
        
        if args.local_rank == 0:
            self.work_dir = get_work_dir(cfg)
            self.logger = get_logger(self.work_dir, cfg)
            cp_projects(cfg.auto_backup, self.work_dir)
        else:
            self.work_dir = None
            self.logger = None
        
        self.best_loss = float('inf')
        self.resume_epoch = self._get_resume_epoch()
        
    def _load_weights(self):
        """Load weights for finetuning or resuming training"""
        if self.cfg.finetune is not None:
            dist_print('Finetuning from:', self.cfg.finetune)
            state_all = torch.load(self.cfg.finetune, map_location='cpu')['model']
            state_clip = {k: v for k, v in state_all.items() if 'model' in k}
            self.net.load_state_dict(state_clip, strict=False)
        
        if self.cfg.resume is not None:
            dist_print('Resuming from:', self.cfg.resume)
            resume_dict = torch.load(self.cfg.resume, map_location='cpu')
            self.net.load_state_dict(resume_dict['model'])
            if 'optimizer' in resume_dict:
                self.optimizer.load_state_dict(resume_dict['optimizer'])
    
    def _get_resume_epoch(self):
        """Get the epoch to resume training from"""
        if self.cfg.resume is not None:
            filename = os.path.basename(self.cfg.resume)
            if filename.startswith('ep') and len(filename) >= 7:
                return int(filename[2:5]) + 1
        return 0
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.net.train()
        progress_bar = dist_tqdm(self.train_loader, desc=f"Epoch {epoch+1:03d}/{self.cfg.epoch}")
        
        epoch_loss = 0.0
        n_batches = len(self.train_loader)
        
        for batch_idx, data_label in enumerate(progress_bar):
            global_step = epoch * n_batches + batch_idx
            
            # Forward pass and loss calculation
            results = inference(self.net, data_label, self.cfg.dataset)
            loss = calc_loss(self.loss_dict, results, self.logger, global_step, epoch)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step(global_step)
            
            # Logging every 10 batches
            if global_step % 10 == 0 and self.logger is not None:
                reset_metrics(self.metric_dict)
                update_metrics(self.metric_dict, results)
                
                # Update progress bar with key metrics only
                if hasattr(progress_bar, 'set_postfix'):
                    key_metrics = {
                        'loss': f'{loss.item():.3f}',
                        'top1': f'{self.metric_dict["op"][0].get():.3f}',
                        'top2': f'{self.metric_dict["op"][1].get():.3f}'
                    }
                    progress_bar.set_postfix(**key_metrics)
            
            epoch_loss += loss.item()
        
        return epoch_loss / n_batches
    
    def run(self):
        """Main training loop"""
        dist_print(f"Starting training at {datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')}")
        dist_print(f"Epochs: {self.cfg.epoch}, Batches per epoch: {len(self.train_loader)}")
        
        for epoch in range(self.resume_epoch, self.cfg.epoch):
            epoch_start_time = time.time()
            
            # Train one epoch
            avg_loss = self.train_epoch(epoch)
            self.train_loader.reset()
            
            # Save checkpoint
            if self.args.local_rank == 0:
                if avg_loss < self.best_loss:
                    self.best_loss = avg_loss
                    save_model(
                        self.net, self.optimizer, epoch, self.work_dir, 
                        self.distributed, filename='best_model.pth'
                    )
                
                # Save periodic checkpoint every 10 epochs
                if (epoch + 1) % 10 == 0:
                    save_model(
                        self.net, self.optimizer, epoch, self.work_dir,
                        self.distributed, filename=f'epoch_{epoch+1:03d}.pth'
                    )
            
            # Log epoch summary
            epoch_time = time.time() - epoch_start_time
            if epoch % 5 == 0 or epoch == self.cfg.epoch - 1:
                dist_print(f"Epoch {epoch+1:03d}/{self.cfg.epoch} - Loss: {avg_loss:.4f} - Time: {epoch_time:.1f}s")
        
        # Final summary
        total_time = time.time() - self.start_time
        dist_print(f"\nTraining completed in {datetime.timedelta(seconds=int(total_time))}")
        dist_print(f"Best loss achieved: {self.best_loss:.4f}")
        
        if self.logger is not None:
            self.logger.close()

def main():
    torch.backends.cudnn.benchmark = True
    
    # Merge configuration
    args, cfg = merge_config()
    
    # Setup distributed training
    distributed = False
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1
    
    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    
    # Initialize and run training session
    training_session = TrainingSession(cfg, args, distributed)
    training_session.run()

if __name__ == "__main__":
    main()
