# Ultra-Fast-Lane-Detection v2 Configuration
# Optimized Fine-tuning for CULane + New Data
# Strategy: Progressive fine-tuning với learning rate scheduling

# ---------- DATA CONFIG ----------
dataset = 'CULane'
data_root = '/content/dataset_14_8'

# ---------- FINE-TUNING STRATEGY ----------
# Phase 1: Warm-up với frozen backbone (epochs 1-15)
# Phase 2: Full fine-tuning với reduced LR (epochs 16-40) 
# Phase 3: Polish training với very low LR (epochs 41-60)
epoch = 60  # Giảm từ 100 để tránh overfitting

# ---------- BATCH SIZE OPTIMIZATION ----------
# Tăng batch_size để stable gradient với data mới
batch_size = 24  # Tăng từ 16, tối ưu cho GPU memory
seed = 42

# ---------- OPTIMIZER CONFIG (Optimized for Fine-tuning) ----------
# SGD thường tốt hơn Adam cho fine-tuning lane detection
optimizer = 'SGD'  # Thay đổi từ Adam về SGD
learning_rate = 0.01  # Giảm 5x từ original (0.05), phù hợp cho fine-tuning
weight_decay = 5e-4   # Tăng regularization để tránh overfitting
momentum = 0.9

# ---------- ADVANCED SCHEDULER CONFIG ----------
# Multi-step với điều chỉnh phù hợp fine-tuning
scheduler = 'multi'
min_lr = 1e-6
steps = [20, 35, 50]  # Điều chỉnh cho 60 epochs
gamma = 0.2  # Giảm LR mạnh hơn (từ 0.1 -> 0.2)

# ---------- WARMUP CONFIG (Critical for Fine-tuning) ----------
warmup = 'linear'
warmup_iters = 200  # Tăng warmup để model thích nghi dần với data mới

# ---------- MODEL CONFIG ----------
backbone = '34'
num_lanes = 4
griding_num = 200
use_aux = False  # Giữ nguyên để tập trung vào main task
fc_norm = True

# ---------- ADVANCED LOSS CONFIG (Fine-tuning Optimized) ----------
# Tăng trọng số cho shape loss để học tốt hơn từ data mới
sim_loss_w = 0.0
shp_loss_w = 1.0     # Tăng từ 0.0, giúp học shape tốt hơn
var_loss_power = 2.0
mean_loss_w = 0.1    # Tăng từ 0.05, ổn định training

# ---------- INPUT CONFIG ----------
num_row = 72
num_col = 81
train_width = 1600
train_height = 320
crop_ratio = 0.8     # Tăng từ 0.6, augmentation mạnh hơn
num_cell_row = 200
num_cell_col = 100

# ---------- LOGGING CONFIG ----------
note = 'finetune_optimized_v2'
log_path = '/content/ufldv2_logs'
auto_backup = True

# ---------- TRANSFER LEARNING CONFIG ----------
finetune = '/content/Ultra-Fast-Lane-Detection-v2/weights/culane_res34.pth'
resume = None

# ---------- TESTING CONFIG ----------
test_model = '/content/Ultra-Fast-Lane-Detection-v2/weights/culane_res34.pth'
test_work_dir = '/content/test_results'
tta = True

# ---------- ADVANCED FINE-TUNING PARAMETERS ----------
# Thêm các tham số cho fine-tuning tối ưu

# Data Augmentation Enhancement
augment_prob = 0.8          # Tăng augmentation probability
color_jitter_prob = 0.5     # Color augmentation cho robustness
rotation_degree = 3         # Nhẹ nhàng rotation
brightness_factor = 0.3     # Brightness variation
contrast_factor = 0.3       # Contrast variation

# Gradient Clipping cho stable training
grad_clip_norm = 10.0       # Gradient clipping value

# Early Stopping Parameters
patience = 8                # Early stopping patience
min_delta = 0.001          # Minimum improvement threshold

# Evaluation During Training
eval_during_training = True # Đánh giá trong quá trình train
eval_interval = 5          # Đánh giá mỗi 5 epochs

# Model EMA (Exponential Moving Average)
use_ema = True             # Sử dụng EMA cho stable model
ema_decay = 0.9999         # EMA decay rate

# Progressive Unfreezing Strategy
freeze_backbone_epochs = 10 # Freeze backbone trong 10 epochs đầu
progressive_unfreeze = True # Gradually unfreeze layers

# Advanced Loss Balancing
focal_loss_alpha = 0.25    # Focal loss alpha
focal_loss_gamma = 2.0     # Focal loss gamma
label_smoothing = 0.1      # Label smoothing factor

# Validation Split for New Data
val_split_ratio = 0.15     # 15% data mới cho validation
shuffle_data = True        # Shuffle data mỗi epoch

# Mixed Precision Training
use_amp = True             # Automatic Mixed Precision
amp_opt_level = 'O1'       # AMP optimization level
