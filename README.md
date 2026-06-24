# 2603 Tester Model (RCNN) - Small Defect Training Pipeline

This folder trains an RCNN-family detector on the same preprocessed dataset flow.

## 1) Candidate RCNN Models for Small Defects

Based on official papers + MMDetection model docs:

1. `TridentNet`
- Focus: scale-aware branches for multi-scale objects.
- Source: ICCV 2019 paper.

2. `Libra R-CNN`
- Focus: balanced sampling/features/loss for training imbalance.
- Source: CVPR 2019 paper.

3. `Dynamic R-CNN`
- Focus: dynamic IoU/loss adaptation for higher-quality boxes with low extra overhead.
- Source: ECCV 2020 paper + official MMDetection config support.

### Default Choice in This Folder

Default model is `Dynamic R-CNN (R50-FPN)` because:
- It is stable with your current stack (`mmdet==3.3.0`, `mmcv==2.1.0`).
- It is strong for high-quality localization on tiled defect data.
- This script also applies small-object anchor scaling (`--anchor-scales 2 4 8`).

You can switch to `Cascade R-CNN` or `Libra Faster R-CNN` using the same script.

## 2) Preprocess (Same as RF-DETR)

```bash
python scripts/prepare_tiled_coco_dataset.py 
  --source-root ./Dataset-v3.v1i.yolov5pytorch 
  --images-subdir train/images 
  --labels-subdir train/labels 
  --secondary-root ./data/dataset_stage2_refined 
  --output-root ./data/rcnn_tiled_coco 
  --val-ratio 0.15 
  --test-ratio 0.10 
  --split-strategy dominant_class 
  --min-defect-side-px 10 
  --pockmark-top-percent 0.10 
  --pockmark-border-px 2 
  --seed 42 
  --overwrite
```

## 3) Train (Recommended Default)

Train Dynamic R-CNN and exclude `unknown` class:

```bash
python scripts/train_mmdet_rcnn.py 
  --dataset-dir ./data/rcnn_tiled_coco 
  --output-dir ./runs/dynamic-rcnn-r50 
  --model dynamic_rcnn_r50_fpn_1x 
  --epochs 36 
  --batch-size 4 
  --num-workers 8 
  --image-size 256 
  --anchor-scales 2 4 8 
  --exclude-classes unknown 
  --amp
```

GPU behavior:
- If `--gpu-id` is omitted, the script auto-picks the least-loaded GPU (by util + VRAM score from `nvidia-smi`).
- Use `--exclude-gpus` to avoid specific devices during auto-pick.

Augmentation behavior (RF-DETR-matched):
- Train pipeline uses RF-DETR-style Albumentations policy:
  - `HorizontalFlip(p=0.2)`
  - `OneOf(RandomBrightnessContrast / ColorJitter / HSV / RGBShift / Gamma / CLAHE / ChannelShuffle / NoOp)`
- At every epoch start, seed is reset as `base_seed + epoch` (same process idea as RF-DETR).
- Train workers are restarted each epoch so worker-side RNG also refreshes.
- Applied augmentation policy is saved to `runs/<run-name>/augmentation_config.json`.
- Disable with `--disable-augment`.

Best-checkpoint behavior:
- Default `save_best` is now `coco/bbox_mAP_class_mean`.
- This metric is computed by a custom evaluator as the mean of per-class AP (`*_precision`) values.
- You can exclude classes from this class-mean with `--class-mean-exclude`.

## 4) Switch to Other RCNN Models

### Cascade R-CNN

```bash
python scripts/train_mmdet_rcnn.py 
  --dataset-dir ./data/rcnn_tiled_coco 
  --output-dir ./runs/cascade-rcnn-r50 
  --model cascade_rcnn_r50_fpn_1x 
  --exclude-classes unknown 
  --amp
```

### Libra Faster R-CNN

```bash
python scripts/train_mmdet_rcnn.py 
  --dataset-dir ./data/rcnn_tiled_coco 
  --output-dir ./runs/libra-faster-rcnn-r50 
  --model libra_faster_rcnn_r50_fpn_1x 
  --exclude-classes unknown 
  --amp
```

## 5) Resume / Test

### Resume Training

```bash
python scripts/train_mmdet_rcnn.py 
  --dataset-dir ./data/rcnn_tiled_coco 
  --output-dir ./runs/dynamic-rcnn-r50 
  --model dynamic_rcnn_r50_fpn_1x 
  --resume-from ./runs/dynamic-rcnn-r50/latest.pth 
  --exclude-classes unknown 
  --amp
```

### Best Metric Control Example

Use class-wise mean AP as best metric (default) while excluding `unknown` from the class-mean:

```bash
python scripts/train_mmdet_rcnn.py 
  --dataset-dir ./data/rcnn_tiled_coco 
  --output-dir ./runs/dynamic-rcnn-r50 
  --model dynamic_rcnn_r50_fpn_1x 
  --exclude-classes unknown 
  --class-mean-exclude unknown 
  --save-best-metric coco/bbox_mAP_class_mean 
  --amp
```

### Test Only

```bash
python scripts/train_mmdet_rcnn.py 
  --dataset-dir ./data/rcnn_tiled_coco 
  --output-dir ./runs/dynamic-rcnn-r50 
  --model dynamic_rcnn_r50_fpn_1x 
  --test-only 
  --load-from ./runs/dynamic-rcnn-r50/latest.pth 
  --exclude-classes unknown
```

### GPU Selection Examples

Auto-select (default):

```bash
python scripts/train_mmdet_rcnn.py --dataset-dir ./data/rcnn_tiled_coco --output-dir ./runs/dynamic-rcnn-r50 --exclude-classes unknown --amp
```

Auto-select but never use GPU 1:

```bash
python scripts/train_mmdet_rcnn.py --dataset-dir ./data/rcnn_tiled_coco --output-dir ./runs/dynamic-rcnn-r50 --exclude-classes unknown --exclude-gpus 1 --amp
```

Force GPU 2:

```bash
python scripts/train_mmdet_rcnn.py --dataset-dir ./data/rcnn_tiled_coco --output-dir ./runs/dynamic-rcnn-r50 --exclude-classes unknown --gpu-id 2 --amp
```

## 6) Official References

- MMDetection Model Zoo: https://mmdetection.readthedocs.io/en/main/model_zoo.html
- MMDetection GitHub: https://github.com/open-mmlab/mmdetection
- Dynamic R-CNN (ECCV 2020): https://arxiv.org/abs/2004.06002
- Libra R-CNN (CVPR 2019): https://arxiv.org/abs/1904.02701
- TridentNet (ICCV 2019): https://openaccess.thecvf.com/content_ICCV_2019/html/Li_Scale-Aware_Trident_Networks_for_Object_Detection_ICCV_2019_paper.html

## 7) Split Visualization (Best Model)

Visualize GT / Prediction / Both using the best checkpoint in run-dir:
- Prediction threshold behavior:
  - Script first tries class-wise best threshold from `./runs/pr_auc_eval/<run-name>/pr_curves_<split>.json`.
  - If split json is missing and split is not `test`, it falls back to `pr_curves_test.json`.
  - If no PR json is found (or class threshold is missing), it falls back to global `--threshold`.
- You can force a specific threshold json with `--class-threshold-json`.

```bash
# train split
python scripts/visualize_coco_bboxes.py 
  --dataset-dir ./data/rcnn_tiled_coco 
  --split train 
  --mode both 
  --run-dir ./runs/dynamic-rcnn-r50 
  --threshold 0.3 
  --class-threshold-json ./runs/pr_auc_eval/dynamic-rcnn-r50/pr_curves_test.json 
  --skip-gt-only-classes unknown 
  --max-images 0 
  --output-dir ./runs/vis/train_both_best

# valid split
python scripts/visualize_coco_bboxes.py 
  --dataset-dir ./data/rcnn_tiled_coco 
  --split valid 
  --mode both 
  --run-dir ./runs/dynamic-rcnn-r50 
  --threshold 0.3 
  --class-threshold-json ./runs/pr_auc_eval/dynamic-rcnn-r50/pr_curves_test.json 
  --skip-gt-only-classes unknown 
  --max-images 0 
  --output-dir ./runs/vis/valid_both_best

# test split
python scripts/visualize_coco_bboxes.py 
  --dataset-dir ./data/rcnn_tiled_coco 
  --split test 
  --mode both 
  --run-dir ./runs/dynamic-rcnn-r50 
  --threshold 0.3 
  --class-threshold-json ./runs/pr_auc_eval/dynamic-rcnn-r50/pr_curves_test.json 
  --skip-gt-only-classes unknown 
  --max-images 0 
  --output-dir ./runs/vis/test_both_best
```


## 8) Class-wise PR Curve / PR-AUC / Best Threshold

Evaluate class-wise PR curves on tiled COCO split and compute:
- PR-AUC
- best threshold per class (max F1)

```bash
python scripts/eval_pr_auc_threshold.py 
  --dataset-dir ./data/rcnn_tiled_coco 
  --split test 
  --run-dir ./runs/dynamic-rcnn-r50 
  --exclude-classes unknown 
  --output-dir ./runs/pr_auc_eval/dynamic-rcnn-r50
```

Optional:
- use specific checkpoint: `--checkpoint ./runs/dynamic-rcnn-r50/latest.pth`
- change IoU threshold: `--iou-threshold 0.5`
- collect denser PR points: `--infer-threshold 0.001`
- draw more detailed PR curve with finer threshold step: `--threshold-step 0.001` (default `0.002`)
- subset test: `--max-images 200`

Outputs:
- `pr_auc_summary_<split>.csv`
- `pr_curves_<split>.json`
- `pr_curves_<split>.png`
- Keep this output under `./runs/pr_auc_eval/<run-name>/` to enable automatic class-wise threshold loading in sections 7 and 9.
- In `pr_auc_summary_<split>.csv`:
  - `num_pred` = number of predictions at the class best-threshold point (best F1).
  - `num_pred_total` = total accumulated predictions used to build the full PR curve.

## 9) Test Confusion Matrix (After Tiling)

Generate confusion matrix on tiled COCO test split:
- Prediction threshold behavior is the same as visualization:
  - class-wise best threshold from `pr_curves_<split>.json` (or `pr_curves_test.json` fallback), else global `--threshold`.
- You can force a specific threshold json with `--class-threshold-json`.

```bash
python scripts/eval_confusion_matrix.py 
  --dataset-dir ./data/rcnn_tiled_coco 
  --split test 
  --run-dir ./runs/dynamic-rcnn-r50 
  --threshold 0.5 
  --class-threshold-json ./runs/pr_auc_eval/dynamic-rcnn-r50/pr_curves_test.json 
  --iou-threshold 0.3 
  --skip-gt-only-classes unknown 
  --max-images 0 
  --output-dir ./runs/confusion/test_best
```

Outputs:
- raw confusion csv/png (with background row/column)
- row-normalized confusion png
- matched-only class confusion csv/png
- summary json
