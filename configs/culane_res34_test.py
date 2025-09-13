# Ultra-Fast-Lane-Detection v2 Test Configuration
# Dataset: CULane + Backbone: ResNet-34

# ---------- DATA CONFIG ----------
dataset = 'CULane'
data_root = '/content/dataset_14_8'

# ---------- TRAINING CONFIG ----------
epoch = 100
batch_size = 16
seed = 42

# ---------- OPTIMIZER CONFIG ----------
optimizer = 'Adam'
learning_rate = 3e-4
weight_decay = 1e-4
momentum = 0.9

# ---------- SCHEDULER CONFIG ----------
scheduler = 'multi'
min_lr = 1e-6
steps = [60, 80, 90]
gamma = 0.1

# ---------- WARMUP CONFIG ----------
warmup = 'linear'
warmup_iters = 150

# ---------- MODEL CONFIG ----------
backbone = '34'
num_lanes = 4
griding_num = 200
use_aux = False
fc_norm = True

# ---------- LOSS CONFIG ----------
sim_loss_w = 0.0
shp_loss_w = 0.0
var_loss_power = 2.0
mean_loss_w = 0.05

# ---------- INPUT CONFIG ----------
num_row = 72
num_col = 81
train_width = 1600
train_height = 320
crop_ratio = 0.6
num_cell_row = 200
num_cell_col = 100

# ---------- LOGGING CONFIG ----------
note = ''
log_path = '/content/ufldv2_logs'
auto_backup = True

# ---------- TRANSFER LEARNING CONFIG ----------
finetune = '/content/Ultra-Fast-Lane-Detection-v2/weights/culane_res34.pth'
resume = None

# ---------- TESTING CONFIG ----------
test_model = '/content/Ultra-Fast-Lane-Detection-v2/weights/culane_res34.pth'
test_work_dir = '/content/test_results'
tta = True

# ---------- EVALUATION CONFIG ----------
eval_mode = 'official'
eval_during_training = False
