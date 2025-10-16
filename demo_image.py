
import os, cv2, torch, argparse, random
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from utils.common import merge_config, get_model
from utils.dist_utils import dist_print

# --- dùng lại logic từ demo_video.py (giữ nguyên tên hàm) ---
def pred2coords(pred, row_anchor, col_anchor, local_width=1,
                original_image_width=1640, original_image_height=590):
    b, num_grid_row, num_cls_row, num_lane_row = pred['loc_row'].shape
    b, num_grid_col, num_cls_col, num_lane_col = pred['loc_col'].shape

    max_indices_row = pred['loc_row'].argmax(1).cpu()
    valid_row       = pred['exist_row'].argmax(1).cpu()

    max_indices_col = pred['loc_col'].argmax(1).cpu()
    valid_col       = pred['exist_col'].argmax(1).cpu()

    pred['loc_row'] = pred['loc_row'].cpu()
    pred['loc_col'] = pred['loc_col'].cpu()

    coords = []
    row_lane_idx = [1, 2]
    col_lane_idx = [0, 3]

    for i in row_lane_idx:
        tmp = []
        if valid_row[0, :, i].sum() > num_cls_row / 2:
            for k in range(valid_row.shape[1]):
                if valid_row[0, k, i]:
                    all_ind = torch.arange(
                        max(0,   max_indices_row[0, k, i] - local_width),
                        min(num_grid_row - 1, max_indices_row[0, k, i] + local_width) + 1
                    )
                    out_tmp = (pred['loc_row'][0, all_ind, k, i].softmax(0) * all_ind.float()).sum() + 0.5
                    out_tmp = out_tmp / (num_grid_row - 1) * original_image_width
                    tmp.append((int(out_tmp), int(row_anchor[k] * original_image_height)))
            coords.append(tmp)

    for i in col_lane_idx:
        tmp = []
        if valid_col[0, :, i].sum() > num_cls_col / 4:
            for k in range(valid_col.shape[1]):
                if valid_col[0, k, i]:
                    all_ind = torch.arange(
                        max(0,   max_indices_col[0, k, i] - local_width),
                        min(num_grid_col - 1, max_indices_col[0, k, i] + local_width) + 1
                    )
                    out_tmp = (pred['loc_col'][0, all_ind, k, i].softmax(0) * all_ind.float()).sum() + 0.5
                    out_tmp = out_tmp / (num_grid_col - 1) * original_image_height
                    tmp.append((int(col_anchor[k] * original_image_width), int(out_tmp)))
            coords.append(tmp)
    return coords

def build_transform(cfg):
    h_resize = int(cfg.train_height / cfg.crop_ratio)
    w_resize = cfg.train_width
    def bottom_crop(pil_img):
        w, h = pil_img.size
        top = max(0, h - cfg.train_height)
        return TF.crop(pil_img, top=top, left=0, height=cfg.train_height, width=w)
    return T.Compose([
        T.Resize((h_resize, w_resize), interpolation=T.InterpolationMode.BILINEAR),
        T.Lambda(bottom_crop),
        T.ToTensor(),
        T.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
    ])

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('config', help='path to config file (vd: configs/culane_res34.py)')
    p.add_argument('--test_model', required=True, help='đường dẫn epXXX.pth để infer')
    p.add_argument('--video', required=True, help='đường dẫn video đầu vào')
    p.add_argument('--out_dir', default='snap_out', help='thư mục lưu ảnh')
    p.add_argument('--num_frames', type=int, default=5, help='số frame cần lấy')
    p.add_argument('--seed', type=int, default=42, help='seed chọn frame ngẫu nhiên')
    p.add_argument('--local_rank', type=int, default=0)
    return p.parse_args()

def draw_lanes(frame_bgr, coords):
    for lane in coords:
        for (x, y) in lane:
            cv2.circle(frame_bgr, (int(x), int(y)), 4, (0,255,0), -1)
    return frame_bgr

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    args = get_args()

    # merge cfg như repo yêu cầu
    import sys as _sys
    _sys.argv = [_sys.argv[0], args.config, '--test_model', args.test_model, '--local_rank', str(args.local_rank)]
    args_cfg, cfg = merge_config()
    cfg.batch_size = 1
    dist_print('start image snapshot inference...')

    # model
    net = get_model(cfg)
    sd = torch.load(args.test_model, map_location='cpu')
    state = sd.get('model', sd)
    new_state = {k.replace('module.', ''): v for k,v in state.items()}
    net.load_state_dict(new_state, strict=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = net.to(device).eval()

    # video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {args.video}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    os.makedirs(args.out_dir, exist_ok=True)

    # chọn frame index
    random.seed(args.seed)
    num = min(args.num_frames, max(1, total))
    # ưu tiên chọn đều + thêm ngẫu nhiên nhẹ để đa dạng
    base_idx = np.linspace(0, total-1, num=num, dtype=int).tolist()
    # đảm bảo unique, sort
    idx_list = sorted(set(base_idx))
    print(f"Will sample frames at indices: {idx_list} / total {total}")

    tfm = build_transform(cfg)
    saved_paths = []

    with torch.no_grad():
        for i, idx in enumerate(idx_list):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame_bgr = cap.read()
            if not ok:
                continue
            img_h, img_w = frame_bgr.shape[:2]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(frame_rgb)
            inp = tfm(pil).unsqueeze(0).to(device)

            pred = net(inp)
            coords = pred2coords(pred, cfg.row_anchor, cfg.col_anchor,
                                 original_image_width=img_w, original_image_height=img_h)
            vis = draw_lanes(frame_bgr.copy(), coords)
            out_path = os.path.join(args.out_dir, f'frame_{idx:06d}.jpg')
            cv2.imwrite(out_path, vis)
            saved_paths.append(out_path)
            print(f'Saved {out_path}')

    cap.release()

    # tạo ảnh ghép 5 khung (nếu đủ ảnh)
    if len(saved_paths) > 0:
        imgs = [cv2.imread(p) for p in saved_paths]
        h, w = imgs[0].shape[:2]
        # ghép theo lưới 1xN (đơn giản, vừa xem nhanh)
        mosaic = cv2.hconcat([cv2.resize(im, (w, h)) for im in imgs])
        mosaic_path = os.path.join(args.out_dir, 'mosaic.jpg')
        cv2.imwrite(mosaic_path, mosaic)
        print(f'Done. Saved mosaic: {mosaic_path}')
    else:
        print('No frames were saved.')
