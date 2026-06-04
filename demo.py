import os
import glob
import torch
import SimpleITK as sitk
import numpy as np
from torchvision import transforms
import argparse

from feature_extract_net import Model_net

def preprocess_image(vol):
    transform = transforms.ToTensor()
    tensor = transform(vol)
    # (H, W, D) -> (D, H, W) 
    tensor = tensor.permute(1, 0, 2).unsqueeze(0).unsqueeze(0)
    return tensor

def decode_output(pred: torch.Tensor, conf_thres: float = 0.5) -> np.ndarray:
    pred = pred.detach().squeeze(0)
    if pred.dim() == 5:
        pred = pred.squeeze(0)
    if pred.dim() != 4 or pred.shape[0] < 4:
        raise ValueError(f"Unexpected pred shape: {tuple(pred.shape)}")

    pred_x, pred_y, pred_z = torch.sigmoid(pred[0]), torch.sigmoid(pred[1]), torch.sigmoid(pred[2])
    conf = torch.sigmoid(pred[3])

    d, h, w = pred_x.shape
    grid_h = torch.arange(h, device=pred.device).view(1, h, 1).expand(d, h, w)
    grid_w = torch.arange(w, device=pred.device).view(1, 1, w).expand(d, h, w)
    grid_d = torch.arange(d, device=pred.device).view(d, 1, 1).expand(d, h, w)

    x = (pred_x + grid_h) * 4.0
    y = (pred_y + grid_w) * 4.0
    z = (pred_z + grid_d) * 4.0

    keep = conf > conf_thres
    if not torch.any(keep):
        return np.empty((0, 4), dtype=np.float32)

    coords_conf = torch.stack([x[keep], y[keep], z[keep], conf[keep]], dim=1)
    return coords_conf.cpu().numpy().astype(np.float32)

def nms_distance(dets: np.ndarray, distance_thres: float = 12.0) -> np.ndarray:
    dets = np.asarray(dets, dtype=np.float32)
    if dets.size == 0:
        return dets

    order = np.argsort(-dets[:, 3])
    dets = dets[order]
    kept = []
    while dets.shape[0] > 0:
        kept.append(dets[0])
        if dets.shape[0] == 1:
            break
        ref = dets[0, :3]
        rest = dets[1:, :3]
        dist = np.sqrt(np.sum((rest - ref) ** 2, axis=1))
        dets = dets[1:][dist > float(distance_thres)]

    return np.stack(kept, axis=0)

def write_swc(points_xyz: np.ndarray, out_path: str,
              resolution_xy: float = 1.0, resolution_z: float = 1.0) -> None:
    """Write (N,3) points to SWC.

    SWC format: id type x y z radius parent
    """
    points_xyz = np.asarray(points_xyz, dtype=np.float32)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Exported soma detections\n")
        f.write("# id type x y z radius parent\n")
        for i, (x, y, z) in enumerate(points_xyz, start=1):
            f.write(
                f"{i} 1 {x * resolution_xy:.3f} {y * resolution_xy:.3f} {z * resolution_z:.3f} 1 -1\n"
            )

def write_csv(points_xyz: np.ndarray, out_path: str,
              resolution_xy: float = 1.0, resolution_z: float = 1.0) -> None:
    """Write detections to CSV with 3 columns (X,Y,Slice) and no header."""
    points_xyz = np.asarray(points_xyz, dtype=np.float32)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for row in points_xyz:
            x = row[0] * resolution_xy
            y = row[1] * resolution_xy
            z = row[2] * resolution_z
            f.write(f"{x:.3f},{y:.3f},{z:.3f}\n")

def main():
    parser = argparse.ArgumentParser(description="PGRMamba (Prior-Guided Region Mamba) Inference Demo")
    parser.add_argument("--weights", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             'checkpoints', 'best_f1_model.pth'),
                        help="Path to the trained model weights")
    parser.add_argument("--input_tif", type=str, default=None,
                        help="Path to the input 3D TIF block")
    parser.add_argument("--conf_thres", type=float, default=0.5,
                        help="Confidence threshold for detection")
    parser.add_argument("--nms_dist_thres", type=float, default=5.0,
                        help="Distance threshold for NMS")
    parser.add_argument("--csv_out_dir", type=str, default=None,
                        help="If set, export prediction points to .csv files in this directory")
    parser.add_argument("--swc_out_dir", type=str, default=None,
                        help="If set, export prediction points to .swc files in this directory")
    parser.add_argument("--resolution_xy", type=float, default=1.0,
                        help="Scale for x/y when writing SWC/CSV")
    parser.add_argument("--resolution_z", type=float, default=1.0,
                        help="Scale for z when writing SWC/CSV")
    args = parser.parse_args()

    
    input_tif = args.input_tif
    if input_tif is None:
        val_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_new", "val")
        tifs = glob.glob(os.path.join(val_dir, "*.tif"))
        if tifs:
            input_tif = tifs[0]  
        else:
            print("❌ No input tif found to run the demo. Please specify --input_tif")
            return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    print(f"Loading Model_net on {device}...")
    model = Model_net([3, 3])  

    if os.path.exists(args.weights):
        state_dict = torch.load(args.weights, map_location=device)
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
        print(f"✅ Weights loaded successfully from: {args.weights}")
    else:
        print(f"⚠️ Warning: Weights not found at {args.weights}. Using random weights.")

    model.to(device).eval()

    
    print(f"Processing sample: {input_tif}")
    image = sitk.ReadImage(input_tif)
    vol = sitk.GetArrayFromImage(image).astype(np.float32)

    vol_tensor = preprocess_image(vol).to(device)

    
    print("Running inference...")
    with torch.no_grad():
        outputs = model(vol_tensor)

    print("Inference completed.")

    
    pred_coords_conf = decode_output(outputs[0], conf_thres=args.conf_thres)
    pred_coords_conf = nms_distance(pred_coords_conf, distance_thres=args.nms_dist_thres)
    pred_coords = (
        pred_coords_conf[:, :3] if pred_coords_conf.size > 0
        else np.empty((0, 3), dtype=np.float32)
    )
    print(f"Detected {len(pred_coords)} soma(s) after NMS.")

    
    name_stem = os.path.splitext(os.path.basename(input_tif))[0]
    if args.csv_out_dir:
        csv_path = os.path.join(args.csv_out_dir, f"{name_stem}.csv")
        write_csv(pred_coords, csv_path,
                  resolution_xy=args.resolution_xy, resolution_z=args.resolution_z)
        print(f"📄 CSV saved to: {csv_path}")

    
    if args.swc_out_dir:
        swc_path = os.path.join(args.swc_out_dir, f"{name_stem}.swc")
        write_swc(pred_coords, swc_path,
                  resolution_xy=args.resolution_xy, resolution_z=args.resolution_z)
        print(f"📄 SWC saved to: {swc_path}")

if __name__ == '__main__':
    main()