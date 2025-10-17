"""
UNet Model Evaluation Script
Evaluates trained UNet model (.h5 weights) on test images for road segmentation.

Usage:
    python eval_unet.py --model_path models/unet_model.h5 --test_dir test_images/
"""

import os
import sys
import argparse
import numpy as np
import cv2
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Add parent directories to path
sys.path.append('../Gistda-Sementi-Segmentation')
sys.path.append('../Gistda-Sementi-Segmentation/Models')
sys.path.append('../Gistda-Sementi-Segmentation/Tools')

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import load_model
    print(f"✅ TensorFlow version: {tf.__version__}")
    
    # Import our custom model loader
    from model_loader import load_model_with_compatibility, print_model_summary
    
except ImportError:
    print("❌ TensorFlow not found. Install with: pip install tensorflow")
    sys.exit(1)

class UNetEvaluator:
    """UNet model evaluator for road segmentation"""
    
    def __init__(self, model_path, input_size=(256, 256), dataset_name="DeepGlobe"):
        """
        Initialize UNet evaluator
        
        Args:
            model_path: Path to .h5 model weights
            input_size: Expected input size (height, width)
            dataset_name: Dataset name for normalization parameters
        """
        self.model_path = model_path
        self.input_size = input_size
        self.dataset_name = dataset_name
        self.model = None
        
        # Load normalization parameters
        self.load_normalization_params()
        
        # Load model
        self.load_model()
    
    def load_normalization_params(self):
        """Load dataset normalization parameters from config"""
        cfg_path = Path("../Gistda-Sementi-Segmentation/cfg.json")
        if cfg_path.exists():
            with open(cfg_path, 'r') as f:
                cfg = json.load(f)
            
            if self.dataset_name in cfg.get("Datasets", {}):
                self.mean = eval(cfg["Datasets"][self.dataset_name]["mean"])
                self.std = eval(cfg["Datasets"][self.dataset_name]["std"])
            else:
                # Default values
                self.mean = [70.95, 71.16, 71.31]
                self.std = [34.00, 35.18, 36.40]
        else:
            # Default values
            self.mean = [70.95, 71.16, 71.31]
            self.std = [34.00, 35.18, 36.40]
        
        print(f"📊 Normalization - Mean: {self.mean}, Std: {self.std}")
    
    def load_model(self):
        """Load UNet model from .h5 file"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        print(f"🔄 Loading model from: {self.model_path}")
        
        try:
            # Use our compatibility loader
            self.model = load_model_with_compatibility(self.model_path, compile=False)
            print_model_summary(self.model)
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def preprocess_image(self, image):
        """
        Preprocess image for model input
        
        Args:
            image: Input image (numpy array)
        
        Returns:
            Preprocessed image tensor
        """
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to input size
        image_resized = cv2.resize(image, self.input_size)
        
        # Normalize
        image_normalized = image_resized.astype(np.float32)
        image_normalized = (image_normalized - np.array(self.mean)) / np.array(self.std)
        
        # Add batch dimension
        image_tensor = np.expand_dims(image_normalized, axis=0)
        
        return image_tensor
    
    def predict(self, image):
        """
        Run prediction on input image
        
        Args:
            image: Input image (numpy array)
        
        Returns:
            Dictionary with segmentation results
        """
        # Preprocess
        image_tensor = self.preprocess_image(image)
        
        # Predict
        prediction = self.model.predict(image_tensor, verbose=0)
        
        # Process output
        # Assuming output is (batch, height, width, num_classes)
        if len(prediction.shape) == 4:
            pred_mask = prediction[0]
            
            # If multi-class, take argmax
            if pred_mask.shape[-1] > 1:
                segmentation_mask = np.argmax(pred_mask, axis=-1)
                probability_map = pred_mask[:, :, 1]  # Road class probability
            else:
                # Binary segmentation
                probability_map = pred_mask[:, :, 0]
                segmentation_mask = (probability_map > 0.5).astype(np.uint8)
        else:
            # Handle different output formats
            probability_map = prediction[0]
            segmentation_mask = (probability_map > 0.5).astype(np.uint8)
        
        return {
            'segmentation_mask': segmentation_mask,
            'probability_map': probability_map,
            'input_shape': image.shape[:2],
            'output_shape': segmentation_mask.shape
        }
    
    def evaluate_image(self, image_path, save_dir=None):
        """
        Evaluate single image
        
        Args:
            image_path: Path to test image
            save_dir: Directory to save results
        
        Returns:
            Evaluation results dictionary
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Predict
        result = self.predict(image)
        
        # Save visualizations if requested
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            image_name = Path(image_path).stem
            
            # Save segmentation mask
            mask_path = save_dir / f"{image_name}_mask.png"
            mask_image = (result['segmentation_mask'] * 255).astype(np.uint8)
            cv2.imwrite(str(mask_path), mask_image)
            
            # Save probability map
            prob_path = save_dir / f"{image_name}_probability.png"
            prob_image = (result['probability_map'] * 255).astype(np.uint8)
            prob_colored = cv2.applyColorMap(prob_image, cv2.COLORMAP_JET)
            cv2.imwrite(str(prob_path), prob_colored)
            
            # Save overlay
            overlay_path = save_dir / f"{image_name}_overlay.png"
            self.save_overlay(image, result['segmentation_mask'], overlay_path)
            
            result['saved_files'] = {
                'mask': str(mask_path),
                'probability': str(prob_path),
                'overlay': str(overlay_path)
            }
        
        return result
    
    def save_overlay(self, image, mask, output_path):
        """Create and save overlay visualization"""
        # Resize mask to match image size if needed
        if image.shape[:2] != mask.shape:
            mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), 
                                     interpolation=cv2.INTER_NEAREST)
        else:
            mask_resized = mask
        
        # Create colored mask
        colored_mask = np.zeros((mask_resized.shape[0], mask_resized.shape[1], 3), dtype=np.uint8)
        colored_mask[mask_resized == 1] = [0, 255, 0]  # Green for roads
        
        # Create overlay
        overlay = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)
        cv2.imwrite(str(output_path), overlay)
    
    def evaluate_directory(self, test_dir, save_dir=None, ground_truth_dir=None):
        """
        Evaluate all images in directory
        
        Args:
            test_dir: Directory containing test images
            save_dir: Directory to save results
            ground_truth_dir: Optional directory with ground truth masks for metrics
        
        Returns:
            Evaluation summary
        """
        test_dir = Path(test_dir)
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        
        # Find all images
        image_files = []
        for ext in image_extensions:
            image_files.extend(test_dir.glob(f"*{ext}"))
        
        if not image_files:
            print(f"❌ No images found in {test_dir}")
            return None
        
        print(f"📁 Found {len(image_files)} images to evaluate")
        
        results = []
        metrics_list = []
        
        for image_path in tqdm(image_files, desc="Evaluating"):
            try:
                # Evaluate image
                result = self.evaluate_image(image_path, save_dir)
                result['image_name'] = image_path.name
                results.append(result)
                
                # Calculate metrics if ground truth available
                if ground_truth_dir:
                    gt_path = Path(ground_truth_dir) / image_path.name
                    if gt_path.exists():
                        metrics = self.calculate_metrics(
                            result['segmentation_mask'],
                            gt_path
                        )
                        metrics['image_name'] = image_path.name
                        metrics_list.append(metrics)
            
            except Exception as e:
                print(f"❌ Error processing {image_path.name}: {e}")
                continue
        
        # Create summary
        summary = {
            'total_images': len(image_files),
            'successful': len(results),
            'failed': len(image_files) - len(results),
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        if metrics_list:
            summary['metrics'] = self.aggregate_metrics(metrics_list)
        
        return summary, results, metrics_list
    
    def calculate_metrics(self, pred_mask, gt_path):
        """Calculate evaluation metrics against ground truth"""
        # Load ground truth
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if gt is None:
            return None
        
        # Resize to match prediction
        if gt.shape != pred_mask.shape:
            gt = cv2.resize(gt, (pred_mask.shape[1], pred_mask.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
        
        # Binarize ground truth
        gt_binary = (gt > 127).astype(np.uint8)
        
        # Calculate metrics
        tp = np.sum((pred_mask == 1) & (gt_binary == 1))
        fp = np.sum((pred_mask == 1) & (gt_binary == 0))
        fn = np.sum((pred_mask == 0) & (gt_binary == 1))
        tn = np.sum((pred_mask == 0) & (gt_binary == 0))
        
        # Precision, Recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # IoU
        intersection = tp
        union = tp + fp + fn
        iou = intersection / union if union > 0 else 0
        
        # Accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'iou': iou,
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn)
        }
    
    def aggregate_metrics(self, metrics_list):
        """Aggregate metrics across all images"""
        if not metrics_list:
            return None
        
        aggregated = {
            'accuracy': np.mean([m['accuracy'] for m in metrics_list]),
            'precision': np.mean([m['precision'] for m in metrics_list]),
            'recall': np.mean([m['recall'] for m in metrics_list]),
            'f1_score': np.mean([m['f1_score'] for m in metrics_list]),
            'iou': np.mean([m['iou'] for m in metrics_list]),
            'std_accuracy': np.std([m['accuracy'] for m in metrics_list]),
            'std_iou': np.std([m['iou'] for m in metrics_list])
        }
        
        return aggregated

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate UNet model on road segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to .h5 model file')
    parser.add_argument('--test_dir', type=str, default='test_images',
                       help='Directory containing test images')
    parser.add_argument('--save_dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--ground_truth_dir', type=str, default=None,
                       help='Directory with ground truth masks (optional)')
    parser.add_argument('--input_size', type=int, nargs=2, default=[256, 256],
                       help='Model input size (height width)')
    parser.add_argument('--dataset', type=str, default='DeepGlobe',
                       choices=['DeepGlobe', 'MassachusettsRoads', 'Spacenet'],
                       help='Dataset name for normalization')
    
    args = parser.parse_args()
    
    print("="*70)
    print("🧪 UNet Model Evaluation")
    print("="*70)
    print(f"📁 Model: {args.model_path}")
    print(f"📁 Test directory: {args.test_dir}")
    print(f"📁 Save directory: {args.save_dir}")
    print(f"📐 Input size: {args.input_size}")
    print("="*70 + "\n")
    
    # Create evaluator
    evaluator = UNetEvaluator(
        model_path=args.model_path,
        input_size=tuple(args.input_size),
        dataset_name=args.dataset
    )
    
    # Run evaluation
    summary, results, metrics = evaluator.evaluate_directory(
        test_dir=args.test_dir,
        save_dir=args.save_dir,
        ground_truth_dir=args.ground_truth_dir
    )
    
    # Save summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = Path(args.save_dir) / f"evaluation_summary_{timestamp}.json"
    
    with open(summary_file, 'w') as f:
        json.dump({
            'summary': summary,
            'metrics': metrics
        }, f, indent=2)
    
    print(f"\n{'='*70}")
    print("📊 EVALUATION SUMMARY")
    print("="*70)
    print(f"Total images: {summary['total_images']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    
    if 'metrics' in summary:
        print(f"\n📈 METRICS (Average):")
        print(f"  Accuracy:  {summary['metrics']['accuracy']:.4f} ± {summary['metrics']['std_accuracy']:.4f}")
        print(f"  Precision: {summary['metrics']['precision']:.4f}")
        print(f"  Recall:    {summary['metrics']['recall']:.4f}")
        print(f"  F1-Score:  {summary['metrics']['f1_score']:.4f}")
        print(f"  IoU:       {summary['metrics']['iou']:.4f} ± {summary['metrics']['std_iou']:.4f}")
    
    print(f"\n💾 Results saved to: {args.save_dir}")
    print(f"📄 Summary saved to: {summary_file}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
