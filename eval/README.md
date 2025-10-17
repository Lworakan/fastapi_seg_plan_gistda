# UNet Model Evaluation

This directory contains tools for evaluating trained UNet models (.h5 weights) on road segmentation tasks.

## 📁 Directory Structure

```
eval/
├── eval_unet.py              # Main evaluation script
├── batch_eval.py             # Batch evaluation for multiple models
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── models/                   # Place your .h5 model files here
│   └── unet_model.h5        # Example model file
│
├── test_images/              # Place test images here
│   ├── test_001.png
│   ├── test_002.png
│   └── ...
│
├── ground_truth/             # Optional: ground truth masks for metrics
│   ├── test_001.png
│   ├── test_002.png
│   └── ...
│
└── results/                  # Evaluation results (auto-generated)
    ├── evaluation_summary_YYYYMMDD_HHMMSS.json
    ├── test_001_mask.png
    ├── test_001_probability.png
    ├── test_001_overlay.png
    └── ...
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data

**Place your model weights:**
```bash
cp /path/to/your/unet_model.h5 eval/models/
```

**Add test images:**
```bash
cp /path/to/test/images/* eval/test_images/
```

**Optional - Add ground truth masks for metrics:**
```bash
cp /path/to/ground_truth/masks/* eval/ground_truth/
```

### 3. Run Evaluation

**Basic evaluation (single model):**
```bash
python eval_unet.py \
  --model_path models/unet_model.h5 \
  --test_dir test_images/ \
  --save_dir results/
```

**With ground truth metrics:**
```bash
python eval_unet.py \
  --model_path models/unet_model.h5 \
  --test_dir test_images/ \
  --save_dir results/ \
  --ground_truth_dir ground_truth/
```

**Custom input size:**
```bash
python eval_unet.py \
  --model_path models/unet_model.h5 \
  --test_dir test_images/ \
  --input_size 512 512 \
  --dataset DeepGlobe
```

### 4. Batch Evaluation (Multiple Models)

Evaluate all models in the `models/` directory:

```bash
python batch_eval.py \
  --model_dir models/ \
  --test_dir test_images/ \
  --save_dir batch_results/ \
  --ground_truth_dir ground_truth/
```

## 📊 Output Files

### Per Image Results

For each test image, the following files are generated:

1. **`{image_name}_mask.png`** - Binary segmentation mask
   - White (255) = Road
   - Black (0) = Background

2. **`{image_name}_probability.png`** - Probability heatmap (JET colormap)
   - Blue = Low probability
   - Red = High probability

3. **`{image_name}_overlay.png`** - Segmentation overlay on original image
   - Green regions = Detected roads

### Summary Files

**`evaluation_summary_{timestamp}.json`**
```json
{
  "summary": {
    "total_images": 50,
    "successful": 48,
    "failed": 2,
    "timestamp": "20251016_120000",
    "metrics": {
      "accuracy": 0.9523,
      "precision": 0.8765,
      "recall": 0.8912,
      "f1_score": 0.8838,
      "iou": 0.7921,
      "std_accuracy": 0.0234,
      "std_iou": 0.0567
    }
  }
}
```

**`model_comparison.csv`** (for batch evaluation)
```csv
Model,Accuracy,Precision,Recall,F1-Score,IoU
unet_epoch10.h5,0.9234,0.8567,0.8712,0.8639,0.7621
unet_epoch20.h5,0.9456,0.8823,0.8956,0.8889,0.8001
unet_best.h5,0.9523,0.8965,0.9012,0.8988,0.8156
```

## 📈 Evaluation Metrics

When ground truth masks are provided, the following metrics are calculated:

| Metric | Description | Range |
|--------|-------------|-------|
| **Accuracy** | Overall pixel accuracy | [0, 1] |
| **Precision** | Positive predictive value | [0, 1] |
| **Recall** | True positive rate / Sensitivity | [0, 1] |
| **F1-Score** | Harmonic mean of precision and recall | [0, 1] |
| **IoU** | Intersection over Union (Jaccard Index) | [0, 1] |

### Formulas

```
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
IoU = TP / (TP + FP + FN)
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

Where:
- TP = True Positives (correctly predicted road pixels)
- FP = False Positives (incorrectly predicted as road)
- FN = False Negatives (road pixels missed)
- TN = True Negatives (correctly predicted non-road)

## 🛠️ Advanced Usage

### Custom Normalization

Edit the normalization parameters in `cfg.json`:

```json
{
  "Datasets": {
    "CustomDataset": {
      "mean": "[123.68, 116.78, 103.94]",
      "std": "[58.40, 57.12, 57.38]"
    }
  }
}
```

Then use:
```bash
python eval_unet.py \
  --model_path models/unet_model.h5 \
  --dataset CustomDataset
```

### Programmatic Usage

```python
from eval_unet import UNetEvaluator

# Initialize evaluator
evaluator = UNetEvaluator(
    model_path='models/unet_model.h5',
    input_size=(256, 256),
    dataset_name='DeepGlobe'
)

# Evaluate single image
result = evaluator.evaluate_image(
    image_path='test_images/test_001.png',
    save_dir='results/'
)

print(f"Segmentation shape: {result['segmentation_mask'].shape}")
print(f"Saved files: {result['saved_files']}")

# Evaluate directory
summary, results, metrics = evaluator.evaluate_directory(
    test_dir='test_images/',
    save_dir='results/',
    ground_truth_dir='ground_truth/'
)

print(f"Average IoU: {summary['metrics']['iou']:.4f}")
```

## 🔧 Troubleshooting

### Issue: Model fails to load

**Error:** `ValueError: Unknown layer`

**Solution:** Load with custom objects:
```python
from tensorflow.keras.models import load_model

custom_objects = {
    'custom_loss': your_custom_loss,
    'custom_metric': your_custom_metric
}

model = load_model('model.h5', custom_objects=custom_objects, compile=False)
```

### Issue: Out of memory

**Solution:** Process images in smaller batches or reduce input size:
```bash
python eval_unet.py --input_size 128 128 ...
```

### Issue: Wrong normalization

**Solution:** Check dataset name matches your training configuration:
```bash
python eval_unet.py --dataset MassachusettsRoads ...
```

## 📝 Notes

1. **Model Format**: Only `.h5` Keras model files are supported
2. **Input Size**: Must match the size used during training
3. **Normalization**: Uses dataset-specific mean/std from `cfg.json`
4. **Ground Truth**: Must have same filename as test images
5. **Color Format**: Images are automatically converted from BGR to RGB

## 🔗 Related Files

- **Training Script**: `../Gistda-Sementi-Segmentation/train_unet.py`
- **Config File**: `../Gistda-Sementi-Segmentation/cfg.json`
- **Model Architecture**: `../Gistda-Sementi-Segmentation/Models/`

## 📚 Examples

### Example 1: Quick Evaluation

```bash
# Single model, no metrics
python eval_unet.py \
  --model_path models/unet_final.h5 \
  --test_dir test_images/
```

### Example 2: Full Evaluation with Metrics

```bash
# With ground truth for metrics calculation
python eval_unet.py \
  --model_path models/unet_best.h5 \
  --test_dir test_images/ \
  --ground_truth_dir ground_truth/ \
  --save_dir results/full_eval/
```

### Example 3: Compare Multiple Models

```bash
# Evaluate all checkpoints
python batch_eval.py \
  --model_dir models/checkpoints/ \
  --test_dir test_images/ \
  --ground_truth_dir ground_truth/ \
  --save_dir comparison_results/
```

### Example 4: Custom Dataset

```bash
# Massachusetts Roads dataset
python eval_unet.py \
  --model_path models/unet_mass.h5 \
  --test_dir test_images/ \
  --dataset MassachusettsRoads \
  --input_size 256 256
```

## 🎯 Best Practices

1. **Always validate** your model on unseen data
2. **Use ground truth** when available for accurate metrics
3. **Compare multiple checkpoints** to find the best model
4. **Check visualizations** before relying solely on metrics
5. **Document** your evaluation parameters and results

## 📊 Interpreting Results

### Good Model Performance
- IoU > 0.75
- F1-Score > 0.85
- Low std deviation across images

### Signs of Issues
- High variation in metrics (large std)
- Good metrics but poor visual quality
- Overfitting: great on training, poor on test

---

**Created on:** Branch `eval-unet-segmentation`  
**Last Updated:** 2025-10-16
