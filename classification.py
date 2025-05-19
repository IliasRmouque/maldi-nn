import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
import argparse

from maldi_nn.models import MaldiTransformer

class MaldiSpectrumDataset(Dataset):
    def __init__(self, mzs, intensities, species=None, locs=None, max_peaks=200):
        """
        Dataset for MALDI spectra
        
        Args:
            mzs: list of arrays with m/z values
            intensities: list of arrays with intensity values
            species: class labels (optional)
            locs: location/index identifiers (optional)
            max_peaks: maximum number of peaks to keep per spectrum
        """
        self.mzs = mzs
        self.intensities = intensities
        self.species = species if species is not None else np.zeros(len(mzs))
        self.locs = locs if locs is not None else np.arange(len(mzs))
        

        
    def __len__(self):
        return len(self.mzs)
    
    def __getitem__(self, idx):
        mz = self.mzs[idx]
        intensity = self.intensities[idx]
        
        
        
        
        return {
            'mz': torch.tensor(mz, dtype=torch.float32),
            'intensity': torch.tensor(intensity, dtype=torch.float32),
            'species': torch.tensor(self.species[idx], dtype=torch.long),
            'loc': self.locs[idx]
        }

def collate_fn(batch):
    """Custom collate function to handle variable length spectra"""
    # Get max length
    max_len = max([len(item['mz']) for item in batch])
    
    # Initialize tensors
    mz = torch.zeros(len(batch), max_len)
    intensity = torch.zeros(len(batch), max_len)
    species = torch.stack([item['species'] for item in batch])
    locs = [item['loc'] for item in batch]
    
    # Fill tensors
    for i, item in enumerate(batch):
        length = len(item['mz'])
        mz[i, :length] = item['mz']
        intensity[i, :length] = item['intensity']
    
    return {
        'mz': mz,
        'intensity': intensity,
        'species': species,
        'loc': locs
    }

def load_data_with_origin_split(data_path, train_ratio=0.8, random_state=42):
    """
    Load data from npz file and split by origin
    
    Args:
        data_path: path to npz file
        train_ratio: ratio of origins to use for training
        random_state: random seed
    
    Returns:
        train_dataset: dataset for training
        val_dataset: dataset for validation
        n_classes: number of unique annotation classes
        class_mapping: dictionary mapping annotation to class index
    """
    # Load data
    data = np.load(data_path, allow_pickle=True)
    mzs = data['mzs']
    intensities = data['spectrums']
    origins = data['origins']
    annotations = data['annotations']


    real_annotations = []
    for i, a in enumerate(annotations):
        if "cck" in a[0].lower():
            real_annotations.append("cck")
        elif "chc" in a[0].lower():
            real_annotations.append("chc")
        elif "fnt" in a[0].lower():
            real_annotations.append("fnt")
        else:
            real_annotations.append("other")


    #remove other
    indices = [i for i, a in enumerate(real_annotations) if a == "other"]
    
    
    #remove the indices from the combined data
    mzs = np.delete(data["mzs"], indices, axis=0)
    intensities = np.delete(data["spectrums"], indices, axis=0)
    annotations = np.delete(real_annotations, indices, axis=0)
    origins = np.delete(data["origins"], indices, axis=0)

    print(f"Loaded {len(mzs)} spectra with {len(np.unique(origins))} unique origins")
    
    # Convert annotations to numeric class indices
    unique_annotations = np.unique(annotations)
    print(f"Found {len(unique_annotations)} unique annotations : {unique_annotations}")
    class_mapping = {ann: idx for idx, ann in enumerate(unique_annotations)}
    numeric_labels = np.array([class_mapping[ann] for ann in annotations])
    
    n_classes = len(unique_annotations)
    print(f"Found {n_classes} unique annotation classes")
    for i, cls in enumerate(unique_annotations):
        count = np.sum(annotations == cls)
        print(f"  Class {i}: {cls} - {count} samples")
    
    # Get unique origins and split
    unique_origins = np.unique(origins)
    np.random.seed(random_state)
    np.random.shuffle(unique_origins)
    split_idx = int(len(unique_origins) * train_ratio)
    train_origins = unique_origins[:split_idx]
    val_origins = unique_origins[split_idx:]
    
    print(f"Training on {len(train_origins)} origins, validating on {len(val_origins)} origins")
    
    # Create masks
    train_mask = np.isin(origins, train_origins)
    val_mask = np.isin(origins, val_origins)
    
    # Create datasets with annotation classes
    train_dataset = MaldiSpectrumDataset(
        mzs=mzs[train_mask],
        intensities=intensities[train_mask],
        species=numeric_labels[train_mask],  # Use annotations as classes
        locs=np.where(train_mask)[0]
    )
    
    val_dataset = MaldiSpectrumDataset(
        mzs=mzs[val_mask],
        intensities=intensities[val_mask],
        species=numeric_labels[val_mask],  # Use annotations as classes
        locs=np.where(val_mask)[0]
    )
    
    print(f"Train set: {len(train_dataset)} spectra, Validation set: {len(val_dataset)} spectra")
    
    return train_dataset, val_dataset, n_classes, class_mapping

def main(args):
    # Load data with origin-based split and get number of classes
    train_dataset, val_dataset, n_classes, class_mapping = load_data_with_origin_split(
        args.data_path, 
        train_ratio=args.train_ratio, 
        random_state=args.seed
    )
    
    # Save the class mapping
    mapping_file = os.path.join(args.output_dir, "class_mapping.json")
    with open(mapping_file, 'w') as f:
        import json
        json.dump({str(k): int(v) for k, v in class_mapping.items()}, f, indent=2)
    print(f"Saved class mapping to {mapping_file}")
    
    # Set number of classes and enable classification
    args.n_classes = n_classes
    args.use_classification = True
    
    print(f"Training with classification enabled: {n_classes} annotation classes")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Initialize model
    model = MaldiTransformer(
        depth=args.depth,
        dim=args.dim,
        n_heads=args.n_heads,
        dropout=args.dropout,
        p=args.masking_probability,
        clf=False,
        n_classes=args.n_classes,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        proportional=args.proportional_masking
    )
    
    # Set up callbacks with additional monitoring for classification
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, "checkpoints"),
        filename="{epoch}-{val_mlmloss:.4f}-{val_clfacc:.4f}",
        monitor= "val_mlmloss",
        mode= "min",
        save_top_k=3,
        save_last=True,
    )
    
    early_stop_callback = EarlyStopping(
        monitor= "val_mlmloss",
        patience=args.patience,
        mode= "min",
        verbose=True,
    )
    
    
    # Set up logger
    logger = TensorBoardLogger(
        save_dir=os.path.join(args.output_dir, "logs"),
        name="maldi_transformer"
    )
    
    # Initialize trainer
    trainer = L.Trainer(
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
    )
    
    # Pre-Train model
    trainer.fit(model, train_loader, val_loader)

    #finetune the model
    # Load the best checkpoint
    best_checkpoint = checkpoint_callback.best_model_path
    print(f"Loading best checkpoint: {best_checkpoint}")
    model = MaldiTransformer.load_from_checkpoint(best_checkpoint)

    model.clf = True
    model.n_classes = args.n_classes
    model.lr = args.learning_rate
    model.weight_decay = args.weight_decay
    model.warmup_steps = args.warmup_steps
    model.proportional = args.proportional_masking
    # Set up a new optimizer and scheduler
    model.configure_optimizers()
    # Set up a new logger
    logger = TensorBoardLogger(
        save_dir=os.path.join(args.output_dir, "logs"),
        name="maldi_transformer_finetune"
    )

    # change early stopping to monitor val_clfacc
    early_stop_callback = EarlyStopping(
        monitor="val_clfacc",
        patience=args.patience,
        mode="max",
        verbose=True,
    )

    # Set up a new trainer for fine-tuning
    finetune_trainer = L.Trainer(
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
    )
    # Fine-tune the model

    finetune_trainer.fit(model, train_loader, val_loader)
    # Save the final model
    final_model_path = os.path.join(args.output_dir, "final_model.ckpt")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MaldiTransformer on custom data")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, default="../data/all_spectra.npz",
                        help="Path to the data file")
    parser.add_argument("--output_dir", type=str, default="../models",
                        help="Directory to save models and logs")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Ratio of origins to use for training")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # DataLoader arguments
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for training and validation")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    
    # Model arguments
    parser.add_argument("--depth", type=int, default=6,
                        help="Depth of the transformer")
    parser.add_argument("--dim", type=int, default=256,
                        help="Dimension of the transformer")
    parser.add_argument("--n_heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout probability")
    parser.add_argument("--masking_probability", type=float, default=0.15,
                        help="Probability of masking tokens for MLM")
    parser.add_argument("--proportional_masking", action="store_true",
                        help="Whether to use proportional masking")
    parser.add_argument("--use_classification", action="store_true",
                        help="Whether to use classification task")
    parser.add_argument("--n_classes", type=int, default=1,
                        help="Number of classes for classification task")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=0.0005,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=2000,
                        help="Number of warmup steps")
    parser.add_argument("--patience", type=int, default=10,
                        help="Patience for early stopping")
    
    args = parser.parse_args()
    #python classification.py --data_path ../data/all_spectra.npz --batch_size 64 --learning_rate 0.0003 --epochs 100 --patience 15
    
   


    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)
    
    main(args)
    

    