import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import lightning as L
import json
import argparse
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from maldi_nn.models import MaldiTransformer
from classification import MaldiSpectrumDataset, collate_fn, load_data_with_origin_split

def main(args):
    # Load class mapping
    with open(os.path.join(args.model_dir, "class_mapping.json"), "r") as f:
        class_mapping = json.load(f)
    
    # Convert string keys back to original format if needed
    if all(k.isdigit() for k in class_mapping.keys()):
        class_mapping = {int(k): v for k, v in class_mapping.items()}
    else:
        class_mapping = {k: v for k, v in class_mapping.items()}
    
    # Inverse mapping for predictions
    idx_to_class = {v: k for k, v in class_mapping.items()}
    n_classes = len(class_mapping)
    
    print(f"Found {n_classes} classes in the model")
    
    # Load data - use the same function as in training
    train_dataset, val_dataset, _, _ = load_data_with_origin_split(
        args.data_path, 
        train_ratio=0.8,  # Use the same split ratio as during training
        random_state=42   # Use the same random seed as during training
    )
    
    # Create data loader with a large batch size for faster inference
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # No need to shuffle for evaluation
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Load the trained model
    checkpoint_path = args.checkpoint_path
    if not checkpoint_path and os.path.exists(os.path.join(args.model_dir, "checkpoints", "last.ckpt")):
        checkpoint_path = os.path.join(args.model_dir, "checkpoints", "last.ckpt")
    elif not checkpoint_path:
        # Find the best checkpoint based on validation accuracy
        checkpoints = [f for f in os.listdir(os.path.join(args.model_dir, "checkpoints")) 
                      if f.endswith(".ckpt") and not f == "last.ckpt"]
        if checkpoints:
            # Sort by validation accuracy (assuming format: epoch-val_mlmloss-val_clfacc.ckpt)
            checkpoints.sort(key=lambda x: float(x.split("-")[-1].replace(".ckpt", "")), reverse=True)
            checkpoint_path = os.path.join(args.model_dir, "checkpoints", checkpoints[0])
    
    print(f"Loading model from {checkpoint_path}")
    
    # Load model with hyperparameters from checkpoint
    model = MaldiTransformer.load_from_checkpoint(checkpoint_path)
    model.eval()  # Set model to evaluation mode
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Prepare for evaluation
    all_predictions = []
    all_labels = []
    
    # Evaluate on training set
    print("Evaluating on training set...")
    with torch.no_grad():
        for batch in tqdm(train_loader):
            # Move inputs to the appropriate device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # Add train_indices for MaldiTransformer's internal processing
            batch["train_indices"] = torch.ones_like(batch["mz"][:, 0], dtype=torch.bool)
            
            # Forward pass
            _, clf_logits = model(batch)
            
            # Get predictions
            predictions = torch.argmax(clf_logits, dim=1).cpu().numpy()
            labels = batch["species"].cpu().numpy()
            
            # Store predictions and labels
            all_predictions.extend(predictions)
            all_labels.extend(labels)
    
    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"Training accuracy: {accuracy:.4f}")
    
    print(np.unique(all_labels, return_counts=True))
    print(np.unique(all_predictions, return_counts=True))
    # Generate detailed classification report
    class_names = [idx_to_class[i] for i in range(n_classes)]
    report = classification_report(
        all_labels, all_predictions, 
        target_names=class_names if isinstance(class_names[0], str) else None,
        digits=4
    )
    print("Classification Report:")
    print(report)
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_names if len(class_names) < 20 else None,
                yticklabels=class_names if len(class_names) < 20 else None)
    plt.title("Confusion Matrix (Training Set)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    
    # Save the confusion matrix
    cm_path = os.path.join(args.model_dir, "train_confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path}")
    
    # Save detailed results
    results = {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": cm.tolist()
    }
    
    results_path = os.path.join(args.model_dir, "train_evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Detailed evaluation results saved to {results_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained MaldiTransformer on training set")
    
    # Data and model arguments
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the data file used for training")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory containing the trained model and class mapping")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to specific model checkpoint (if not provided, uses best or last checkpoint)")
    
    # Evaluation arguments
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    
    args = parser.parse_args()
    
    main(args)