import torch
import torch.nn as nn
import os
import datetime
import time
import numpy as np
from collections import defaultdict
from contextlib import contextmanager
import json

from utils.dist_utils import dist_print, dist_tqdm, synchronize
from utils.factory import get_metric_dict, get_loss_dict, get_optimizer, get_scheduler
from utils.metrics import update_metrics, reset_metrics
from utils.common import calc_loss, get_model, get_train_loader, inference, merge_config, save_model, cp_projects
from utils.common import get_work_dir, get_logger
from evaluation.eval_wrapper import eval_lane

@contextmanager
def timer(name):
    """Context manager for timing operations"""
    start = time.time()
    yield
    dist_print(f"[TIMER] {name}: {time.time() - start:.2f}s")

class EarlyStopping:
    """Early stopping utility for preventing overfitting"""
    def __init__(self, patience=8, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'max':
            if score < self.best_score + self.min_delta:
                self.counter += 1
            else:
                self.best_score = score
                self.counter = 0
        else:  # mode == 'min'
            if score > self.best_score - self.min_delta:
                self.counter += 1
            else:
                self.best_score = score
                self.counter = 0
                
        if self.counter >= self.patience:
            self.early_stop = True
        return self.early_stop

class GradientClipping:
    """Advanced gradient clipping for stable training"""
    def __init__(self, max_norm=1.0, norm_type=2):
        self.max_norm = max_norm
        self.norm_type = norm_type
        
    def __call__(self, model):
        return torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_norm, self.norm_type)

class ModelEMA:
    """Exponential Moving Average for model parameters"""
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()
        
    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
                
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
                
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

class AdvancedTrainingSession:
    """Advanced training session with fine-tuning optimizations"""
    
    def __init__(self, cfg, args, distributed):
        self.cfg = cfg
        self.args = args
        self.distributed = distributed
        self.start_time = time.time()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_score = 0.0
        self.training_log = defaultdict(list)
        
        # Setup components
        self._setup_model_and_data()
        self._setup_training_components()
        self._setup_advanced_features()
        
        # Load weights
        self.resume_epoch = self._load_weights()
        
        dist_print(f"[INFO] Training initialized. Resume from epoch: {self.resume_epoch}")
        
    def _setup_model_and_data(self):
        """Setup model and data loader"""
        with timer("Model and data setup"):
            self.train_loader = get_train_loader(self.cfg)
            self.net = get_model(self.cfg)
            
            if self.distributed:
                self.net = torch.nn.parallel.DistributedDataParallel(
                    self.net, device_ids=[self.args.local_rank], 
                    find_unused_parameters=True  # For fine-tuning flexibility
                )
    
    def _setup_training_components(self):
        """Setup optimizer, scheduler, loss, and metrics"""
        with timer("Training components setup"):
            self.optimizer = get_optimizer(self.net, self.cfg)
            self.scheduler = get_scheduler(self.optimizer, self.cfg, len(self.train_loader))
            self.metric_dict = get_metric_dict(self.cfg)
            self.loss_dict = get_loss_dict(self.cfg)
            
            # Setup logging
            if self.args.local_rank == 0:
                self.work_dir = get_work_dir(self.cfg)
                self.logger = get_logger(self.work_dir, self.cfg)
                cp_projects(self.cfg.auto_backup, self.work_dir)
            else:
                self.work_dir = None
                self.logger = None
    
    def _setup_advanced_features(self):
        """Setup advanced training features"""
        # Early stopping
        if hasattr(self.cfg, 'patience'):
            self.early_stopping = EarlyStopping(
                patience=getattr(self.cfg, 'patience', 8),
                min_delta=getattr(self.cfg, 'min_delta', 0.001)
            )
        else:
            self.early_stopping = None
            
        # Gradient clipping
        if hasattr(self.cfg, 'grad_clip_norm'):
            self.grad_clipper = GradientClipping(max_norm=self.cfg.grad_clip_norm)
        else:
            self.grad_clipper = None
            
        # Model EMA
        if getattr(self.cfg, 'use_ema', False):
            self.model_ema = ModelEMA(
                self.net.module if self.distributed else self.net,
                decay=getattr(self.cfg, 'ema_decay', 0.9999)
            )
        else:
            self.model_ema = None
            
        # Mixed precision training
        self.use_amp = getattr(self.cfg, 'use_amp', False)
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            
    def _load_weights(self):
        """Load weights for fine-tuning or resuming"""
        resume_epoch = 0
        
        if self.cfg.finetune is not None:
            dist_print(f'[FINETUNE] Loading pretrained weights from: {self.cfg.finetune}')
            try:
                checkpoint = torch.load(self.cfg.finetune, map_location='cpu')
                state_dict = checkpoint.get('model', checkpoint)
                
                # Handle different checkpoint formats
                model_state = {}
                for k, v in state_dict.items():
                    if k.startswith('module.'):
                        model_state[k[7:]] = v
                    else:
                        model_state[k] = v
                
                # Progressive loading strategy
                if getattr(self.cfg, 'progressive_unfreeze', False):
                    self._progressive_load(model_state)
                else:
                    self.net.load_state_dict(model_state, strict=False)
                    
                dist_print('[FINETUNE] Pretrained weights loaded successfully')
                
            except Exception as e:
                dist_print(f'[ERROR] Failed to load finetune weights: {e}')
                raise
        
        if self.cfg.resume is not None:
            dist_print(f'[RESUME] Resuming training from: {self.cfg.resume}')
            try:
                checkpoint = torch.load(self.cfg.resume, map_location='cpu')
                self.net.load_state_dict(checkpoint['model'])
                
                if 'optimizer' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                    
                if 'epoch' in checkpoint:
                    resume_epoch = checkpoint['epoch'] + 1
                else:
                    # Extract epoch from filename
                    filename = os.path.basename(self.cfg.resume)
                    if filename.startswith('ep') and len(filename) >= 7:
                        resume_epoch = int(filename[2:5]) + 1
                        
                dist_print(f'[RESUME] Training resumed from epoch {resume_epoch}')
                
            except Exception as e:
                dist_print(f'[ERROR] Failed to resume training: {e}')
                raise
                
        return resume_epoch
    
    def _progressive_load(self, state_dict):
        """Progressive loading for fine-tuning"""
        model_dict = self.net.state_dict()
        
        # Load backbone first
        backbone_dict = {k: v for k, v in state_dict.items() 
                        if 'backbone' in k and k in model_dict}
        model_dict.update(backbone_dict)
        
        # Load neck if compatible
        neck_dict = {k: v for k, v in state_dict.items() 
                    if 'neck' in k and k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(neck_dict)
        
        self.net.load_state_dict(model_dict)
        dist_print(f'[PROGRESSIVE] Loaded {len(backbone_dict)} backbone + {len(neck_dict)} neck parameters')
        
    def _freeze_layers(self, epoch):
        """Progressive layer unfreezing strategy"""
        if not getattr(self.cfg, 'progressive_unfreeze', False):
            return
            
        freeze_epochs = getattr(self.cfg, 'freeze_backbone_epochs', 10)
        
        if epoch < freeze_epochs:
            # Freeze backbone
            for name, param in self.net.named_parameters():
                if 'backbone' in name:
                    param.requires_grad = False
        else:
            # Unfreeze all layers
            for param in self.net.parameters():
                param.requires_grad = True
                
    def _train_epoch(self, epoch):
        """Train one epoch with advanced features"""
        self.net.train()
        self._freeze_layers(epoch)
        
        # Statistics
        epoch_loss = 0.0
        epoch_metrics = defaultdict(list)
        
        progress_bar = dist_tqdm(
            self.train_loader, 
            desc=f"[E{epoch+1:03d}/{self.cfg.epoch:03d}]"
        )
        
        for batch_idx, data_label in enumerate(progress_bar):
            self.global_step = epoch * len(self.train_loader) + batch_idx
            
            # Forward pass
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    results = inference(self.net, data_label, self.cfg.dataset)
                    loss = calc_loss(self.loss_dict, results, self.logger, self.global_step, epoch)
            else:
                results = inference(self.net, data_label, self.cfg.dataset)
                loss = calc_loss(self.loss_dict, results, self.logger, self.global_step, epoch)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.grad_clipper:
                    self.scaler.unscale_(self.optimizer)
                    self.grad_clipper(self.net)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clipper:
                    self.grad_clipper(self.net)
                self.optimizer.step()
            
            self.scheduler.step(self.global_step)
            
            # Update EMA
            if self.model_ema:
                self.model_ema.update()
            
            # Statistics
            epoch_loss += loss.item()
            
            # Logging and metrics
            if self.global_step % 20 == 0:
                reset_metrics(self.metric_dict)
                update_metrics(self.metric_dict, results)
                
                # Log to tensorboard
                if self.logger:
                    for name, op in zip(self.metric_dict['name'], self.metric_dict['op']):
                        self.logger.add_scalar(f'train/{name}', op.get(), self.global_step)
                    self.logger.add_scalar('meta/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
                
                # Update progress bar
                if hasattr(progress_bar, 'set_postfix'):
                    postfix = {'loss': f'{loss.item():.4f}'}
                    for name, op in zip(self.metric_dict['name'][:3], self.metric_dict['op'][:3]):
                        if 'lane' not in name.lower():
                            postfix[name[:6]] = f'{op.get():.3f}'
                    progress_bar.set_postfix(**postfix)
        
        return epoch_loss / len(self.train_loader)
    
    def _evaluate(self, epoch):
        """Evaluate model performance"""
        if not getattr(self.cfg, 'eval_during_training', False):
            return None
            
        eval_interval = getattr(self.cfg, 'eval_interval', 5)
        if epoch % eval_interval != 0 and epoch != self.cfg.epoch - 1:
            return None
            
        dist_print(f"[EVAL] Evaluating at epoch {epoch + 1}")
        
        # Use EMA model for evaluation if available
        if self.model_ema:
            self.model_ema.apply_shadow()
            
        try:
            with timer("Evaluation"):
                result = eval_lane(self.net, self.cfg, ep=epoch, logger=self.logger)
        finally:
            # Restore original model
            if self.model_ema:
                self.model_ema.restore()
                
        return result
    
    def _save_checkpoint(self, epoch, score, avg_loss):
        """Save model checkpoint"""
        if self.args.local_rank != 0:
            return
            
        # Save best model
        if score is not None and score > self.best_score:
            self.best_score = score
            save_model(self.net, self.optimizer, epoch, self.work_dir, self.distributed)
            
            # Save additional info
            checkpoint_info = {
                'epoch': epoch,
                'score': score,
                'loss': avg_loss,
                'config': vars(self.cfg),
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            with open(os.path.join(self.work_dir, 'best_checkpoint_info.json'), 'w') as f:
                json.dump(checkpoint_info, f, indent=2)
                
            dist_print(f"[SAVE] New best model saved! Score: {score:.4f}")
        
        # Periodic saves
        if (epoch + 1) % 10 == 0:
            filename = f'checkpoint_ep{epoch+1:03d}.pth'
            save_path = os.path.join(self.work_dir, filename)
            
            checkpoint = {
                'model': self.net.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epoch': epoch,
                'score': score,
                'loss': avg_loss,
                'config': vars(self.cfg)
            }
            
            torch.save(checkpoint, save_path)
            dist_print(f"[SAVE] Checkpoint saved: {filename}")
    
    def run(self):
        """Main training loop"""
        dist_print("="*80)
        dist_print(f"[START] Advanced Fine-tuning Training Session")
        dist_print(f"[CONFIG] Epochs: {self.cfg.epoch}, Batch size: {self.cfg.batch_size}")
        dist_print(f"[CONFIG] Learning rate: {self.cfg.learning_rate}, Optimizer: {self.cfg.optimizer}")
        dist_print(f"[CONFIG] Model: {self.cfg.backbone}, Dataset: {self.cfg.dataset}")
        dist_print("="*80)
        
        try:
            for epoch in range(self.resume_epoch, self.cfg.epoch):
                epoch_start = time.time()
                
                # Train epoch
                avg_loss = self._train_epoch(epoch)
                self.train_loader.reset()
                
                # Evaluate
                score = self._evaluate(epoch)
                
                # Save checkpoint
                self._save_checkpoint(epoch, score, avg_loss)
                
                # Early stopping
                if self.early_stopping and score is not None:
                    if self.early_stopping(score):
                        dist_print(f"[EARLY STOP] Training stopped at epoch {epoch + 1}")
                        break
                
                # Log epoch summary
                epoch_time = time.time() - epoch_start
                if epoch % 5 == 0 or epoch == self.cfg.epoch - 1:
                    summary = f"[E{epoch+1:03d}] Loss: {avg_loss:.4f}, Time: {epoch_time:.1f}s"
                    if score is not None:
                        summary += f", Score: {score:.4f} (Best: {self.best_score:.4f})"
                    dist_print(summary)
                
                # Store training log
                self.training_log['epoch'].append(epoch + 1)
                self.training_log['loss'].append(avg_loss)
                self.training_log['time'].append(epoch_time)
                if score is not None:
                    self.training_log['score'].append(score)
                    
        except KeyboardInterrupt:
            dist_print("[INTERRUPT] Training interrupted by user")
        except Exception as e:
            dist_print(f"[ERROR] Training failed: {e}")
            raise
        finally:
            self._finalize_training()
    
    def _finalize_training(self):
        """Finalize training session"""
        total_time = time.time() - self.start_time
        dist_print("="*80)
        dist_print(f"[COMPLETE] Training finished!")
        dist_print(f"[SUMMARY] Total time: {datetime.timedelta(seconds=int(total_time))}")
        dist_print(f"[SUMMARY] Best score: {self.best_score:.4f}")
        
        if self.args.local_rank == 0:
            # Save training log
            log_path = os.path.join(self.work_dir, 'training_log.json')
            with open(log_path, 'w') as f:
                json.dump(dict(self.training_log), f, indent=2)
            dist_print(f"[LOG] Training log saved to: {log_path}")
            
        if self.logger:
            self.logger.close()
            
        dist_print("="*80)

def main():
    """Main function"""
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False  # For speed
    
    # Configuration
    args, cfg = merge_config()
    
    # Distributed setup
    distributed = False
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1
        
    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    
    # Synchronize before starting
    synchronize()
    cfg.distributed = distributed
    
    # Validate configuration
    assert cfg.backbone in ['18', '34', '50', '101', '152', '50next', '101next', '50wide', '101wide', '34fca'], \
           f"Unsupported backbone: {cfg.backbone}"
    
    # Initialize and run training
    training_session = AdvancedTrainingSession(cfg, args, distributed)
    training_session.run()

if __name__ == "__main__":
    main()
