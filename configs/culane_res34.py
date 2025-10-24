# Ultra-Fast-Lane-Detection v2 Configuration
# Dataset: CULane + Backbone: ResNet-34 (fine-tune)

# ---------- DATA CONFIG ----------
dataset = 'CULane'
data_root = '/content/Custom_Dataset'

# ---------- TRAINING CONFIG ----------
epoch = 100
batch_size = 16
seed = 42

# ---------- OPTIMIZER CONFIG ----------
optimizer = 'SGD'
learning_rate = 0.0125   # (batch_size=16; gốc 0.025 cho batch 32 → scale)
momentum = 0.9
weight_decay = 1e-4

# ---------- SCHEDULER CONFIG ----------
scheduler = 'multi'
min_lr = 1e-6
steps = [60, 80, 90]
gamma = 0.1

# ---------- WARMUP CONFIG ----------
warmup = 'linear'
warmup_iters = 1000


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
finetune = '/content/weights/culane_res34.pth'
resume = None

# ---------- TESTING CONFIG ----------
test_model = ''
test_work_dir = ''
tta = True

# ---------- EARLY STOP CONFIG ----------
early_stop = True
es_patience = 10
es_min_delta = 1e-3
es_warmup_epochs = 7
