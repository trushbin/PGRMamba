# PGRMamba (Prior-Guided Region Mamba)

PGRMamba is a 3D neuron detection (soma detection) inference demo based on the Mamba architecture. It takes 3D morphological image blocks (.tif) as input and outputs detected soma 3D coordinates in CSV and SWC formats.

## 🔥 Quick Start

### Install Dependencies

```bash
cd /PGRMamba
pip install -r requirements.txt

# Compile Mamba SSM CUDA kernels (required for first-time use)
cd mamba && pip install -e . && cd ..
```

### Run the Demo

Without any arguments, the script automatically picks the first .tif file from `../data_new/val/` as input:

```bash
python demo.py
```

Specify an input image and export CSV/SWC files:

```bash
python demo.py \
    --input_tif ../3D-NSD/val/Soma_001_0000.tif \
    --csv_out_dir ./results/csv \
    --swc_out_dir ./results/swc
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--weights` | str | `./checkpoints/best_f1_model.pth` | Path to model weights |
| `--input_tif` | str | Auto-selects first `3D-NSD/val/*.tif` | Input 3D TIF image block |
| `--conf_thres` | float | 0.5 | Confidence threshold for detection |
| `--nms_dist_thres` | float | 5.0 | NMS distance threshold (in pixels) |
| `--csv_out_dir` | str | None | Output directory for CSV files |
| `--swc_out_dir` | str | None | Output directory for SWC files |
| `--resolution_xy` | float | 1.0 | Scale factor for X/Y coordinates |
| `--resolution_z` | float | 1.0 | Scale factor for Z coordinates |

## 📂 Outputs

- **CSV files**: 3-column headerless format `X, Y, Slice`, consistent with training annotation format.
- **SWC files**: Standard SWC format `id type x y z radius parent`, with each detection as an isolated node (parent=-1).

## 📁 Directory Structure

```
PGRMamba/
├── demo.py                  # Main inference script
├── feature_extract_net.py   # Network architecture (Model_net)
├── mackernel.py             # 3D max-pooling helper module
├── mamba/                   # Mamba SSM sub-module
│   ├── mamba.py             # G_SAM / ChanceMamba / Conv3D definitions
│   └── mamba_ssm/           # Mamba SSM core operators
├── checkpoints/
│   └── best_f1_model.pth    # Pre-trained weights
├── results/                 # Output directory
└── requirements.txt         # Python dependencies
```

##  Example

```bash
# Run inference on a single image with CSV and SWC output
python demo.py \
    --input_tif ../3D-NSD/val/Soma_001_0000.tif \
    --csv_out_dir ./results/csv \
    --swc_out_dir ./results/swc \
    --conf_thres 0.5 \
    --nms_dist_thres 5.0
```

Example output:
```
Loading Model_net on cuda...
✅ Weights loaded successfully from: ./checkpoints/best_f1_model.pth
Processing sample: ../3D-NSD/val/Soma_001_0000.tif
Running inference...
Inference completed.
Detected 3 soma(s) after NMS.
📄 CSV saved to: ./results/csv/Soma_001_0000.csv
📄 SWC saved to: ./results/swc/Soma_001_0000.swc
```

## 📄 License

This project is released under the MIT License.