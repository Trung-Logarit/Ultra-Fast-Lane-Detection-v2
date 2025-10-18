# demo_video2_full.py
import os, cv2, torch, argparse, json
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from utils.common import merge_config, get_model
from utils.dist_utils import dist_print

# ==== logits -> tọa độ ====
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
    row_lane_idx = [1, 2]   # theo UFLDv2
    col_lane_idx = [0, 3]

    # row branch
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
            if tmp: coords.append(tmp)

    # col branch
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
            if tmp: coords.append(tmp)
    return coords

# ==== argparse ====
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('config', help='đường dẫn config, ví dụ: configs/culane_res34.py')
    p.add_argument('--test_model', required=True)
    p.add_argument('--video', required=True)
    p.add_argument('--out', default='out.avi')
    p.add_argument('--mask_dir', default=None)
    p.add_argument('--mask_thickness', type=int, default=6)
    p.add_argument('--log_jsonl', default=None)
    p.add_argument('--local_rank', type=int, default=0)
    return p.parse_args()

# ==== transform ====
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

# ==== draw lanes mask ====
def draw_lanes_to_mask(mask, lanes, thickness=6):
    for lane in lanes:
        if len(lane) >= 2:
            pts = np.array(lane, np.int32).reshape(-1, 1, 2)
            cv2.polylines(mask, [pts], False, 255, thickness, cv2.LINE_AA)

# ==== main ====
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    args = get_args()

    # load cfg + model
    import sys
    sys.argv = [sys.argv[0], args.config, '--test_model', args.test_model, '--local_rank', str(args.local_rank)]
    args_cfg, cfg = merge_config()
    cfg.batch_size = 1
    dist_print('start video inference...')

    assert cfg.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide','34fca']
    net = get_model(cfg)
    sd = torch.load(args.test_model, map_location='cpu')
    state = sd['model'] if 'model' in sd else sd
    new_state = {k.replace('module.', ''): v for k, v in state.items()}
    net.load_state_dict(new_state, strict=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = net.to(device).eval()

    # video io
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {args.video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps): fps = 30.0
    out_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    vout = cv2.VideoWriter(args.out, fourcc, fps, (out_w, out_h))

    # mask + log
    if args.mask_dir:
        os.makedirs(args.mask_dir, exist_ok=True)
    log_f = open(args.log_jsonl, 'w', encoding='utf-8') if args.log_jsonl else None

    tfm = build_transform(cfg)
    frame_idx = 0
    print(f"🎞️ Tổng số frame trong video: {total_frames}")

    with torch.no_grad():
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                print("⏹️ Hết video hoặc lỗi đọc frame.")
                break
            frame_idx += 1
            img_h, img_w = frame_bgr.shape[:2]

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(frame_rgb)
            inp = tfm(pil).unsqueeze(0).to(device)

            pred = net(inp)
            coords = pred2coords(pred, cfg.row_anchor, cfg.col_anchor,
                                 original_image_width=img_w,
                                 original_image_height=img_h)

            # Vẽ overlay
            for lane in coords:
                for (x, y) in lane:
                    cv2.circle(frame_bgr, (int(x), int(y)), 4, (0, 255, 0), -1)

            # Ghi mask
            if args.mask_dir:
                mask = np.zeros((img_h, img_w), np.uint8)
                draw_lanes_to_mask(mask, coords, thickness=args.mask_thickness)
                cv2.imwrite(os.path.join(args.mask_dir, f"{frame_idx:06d}.png"), mask)

            # Ghi log
            if log_f:
                log_f.write(json.dumps({
                    "frame": frame_idx,
                    "width": img_w,
                    "height": img_h,
                    "lanes": coords
                }, ensure_ascii=False) + "\n")
                # ✅ Flush định kỳ để không mất log
                if frame_idx % 10 == 0:
                    log_f.flush()

            # Ghi video
            vout.write(frame_bgr)

            if frame_idx % 50 == 0:
                print(f"✅ Processed {frame_idx}/{total_frames} frames...")

    cap.release(); vout.release()
    if log_f:
        log_f.flush()
        log_f.close()

    # ==== Kiểm tra log đầy đủ ====
    if args.log_jsonl and os.path.exists(args.log_jsonl):
        with open(args.log_jsonl, 'r', encoding='utf-8') as f:
            n_log = sum(1 for _ in f if _.strip())
        print(f"\n📊 Log JSONL chứa {n_log} frame / tổng {total_frames} frame video.")
        if abs(n_log - total_frames) <= 1:
            print("✅ Log đầy đủ 100% khung hình.")
        else:
            print("⚠️ Cảnh báo: log thiếu một phần cuối (video dài hoặc đọc lỗi).")

    print(f"\n🎬 Video xuất: {args.out}")
    if args.mask_dir: print(f"🧩 Mask lane tại: {args.mask_dir}")
    if args.log_jsonl: print(f"📝 Log lane: {args.log_jsonl}")
