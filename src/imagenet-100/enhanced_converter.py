"""
Enhanced PyTorch dataset saver with better memory management and metadata support.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Resize
from PIL import Image
from io import BytesIO
import pandas as pd
import pathlib
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple
import os
import json
from dataclasses import dataclass


@dataclass
class DatasetMetadata:
    """Metadata for saved PyTorch datasets."""
    num_samples: int
    image_size: Tuple[int, int]
    num_classes: int
    mean: List[float]
    std: List[float]
    original_split: str
    source_files: List[str]
    created_at: str


class ParquetToTorchConverter:
    """Enhanced converter with chunking and metadata support."""
    
    def __init__(self, 
                 image_size: Tuple[int, int] = (224, 224),
                 chunk_size: int = 1000,
                 save_metadata: bool = True):
        self.image_size = image_size
        self.chunk_size = chunk_size
        self.save_metadata = save_metadata
        self.transform = Compose([Resize(image_size), ToTensor()])
        
    def _process_chunk(self, chunk_data: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process a chunk of data efficiently."""
        images = []
        labels = []
        
        for _, row in chunk_data.iterrows():
            try:
                img_bytes = row['image']['bytes']
                image = Image.open(BytesIO(img_bytes)).convert('RGB')
                image_tensor = self.transform(image)
                label = torch.tensor(row['label'], dtype=torch.long)
                
                images.append(image_tensor)
                labels.append(label)
            except Exception as e:
                print(f"Warning: Skipping corrupted image: {e}")
                continue
                
        return torch.stack(images), torch.stack(labels)
    
    def _compute_statistics(self, images: torch.Tensor) -> Tuple[List[float], List[float]]:
        """Compute dataset mean and std for normalization."""
        # Normalize to [0, 1] if not already
        if images.max() > 1.0:
            images = images / 255.0
            
        mean = images.mean(dim=[0, 2, 3]).tolist()
        std = images.std(dim=[0, 2, 3]).tolist()
        return mean, std
    
    def convert_and_save(self, 
                        parquet_files: List[pathlib.Path], 
                        output_dir: str,
                        split: str,
                        save_individual_chunks: bool = True) -> Dict[str, Any]:
        """Convert parquet files to PyTorch format with enhanced features."""
        os.makedirs(output_dir, exist_ok=True)
        
        all_images = []
        all_labels = []
        source_files = []
        
        # Process each parquet file
        for pq_file in tqdm(parquet_files, desc=f"Processing {split} files"):
            print(f"Loading {pq_file}")
            data = pd.read_parquet(pq_file)
            source_files.append(str(pq_file))
            
            # Process in chunks to manage memory
            for i in range(0, len(data), self.chunk_size):
                chunk = data.iloc[i:i+self.chunk_size]
                images, labels = self._process_chunk(chunk)
                
                all_images.append(images)
                all_labels.append(labels)
                
                # Save individual chunks if requested
                if save_individual_chunks:
                    chunk_idx = i // self.chunk_size
                    filename = f"{split}_chunk_{chunk_idx:04d}.pt"
                    filepath = os.path.join(output_dir, filename)
                    
                    torch.save({
                        "images": images,
                        "labels": labels,
                        "metadata": {
                            "chunk_idx": chunk_idx,
                            "split": split,
                            "num_samples": len(images)
                        }
                    }, filepath)
        
        # Combine all data
        final_images = torch.cat(all_images, dim=0)
        final_labels = torch.cat(all_labels, dim=0)
        
        # Compute statistics
        mean, std = self._compute_statistics(final_images)
        
        # Create metadata
        metadata = DatasetMetadata(
            num_samples=len(final_images),
            image_size=self.image_size,
            num_classes=int(final_labels.max().item()) + 1,
            mean=mean,
            std=std,
            original_split=split,
            source_files=source_files,
            created_at=str(pd.Timestamp.now())
        )
        
        # Save complete dataset
        complete_file = os.path.join(output_dir, f"{split}_complete.pt")
        torch.save({
            "images": final_images,
            "labels": final_labels,
            "metadata": metadata
        }, complete_file)
        
        # Save metadata separately for easy access
        if self.save_metadata:
            metadata_file = os.path.join(output_dir, f"{split}_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata.__dict__, f, indent=2)
        
        print(f"Saved {split} dataset: {len(final_images)} samples to {complete_file}")
        print(f"Dataset stats - Mean: {mean}, Std: {std}")
        
        return {
            "file": complete_file,
            "metadata": metadata,
            "num_samples": len(final_images)
        }


class ImageNet100Dataset(Dataset):
    """PyTorch Dataset class for easy loading of saved ImageNet-100 data."""
    
    def __init__(self, data_file: str, transform: Optional[Any] = None):
        self.data = torch.load(data_file)
        self.images = self.data["images"]
        self.labels = self.data["labels"]
        self.metadata = self.data.get("metadata", {})
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
    
    def get_statistics(self):
        """Get dataset normalization statistics."""
        return (
            self.metadata.get("mean", [0.485, 0.456, 0.406]),
            self.metadata.get("std", [0.229, 0.224, 0.225])
        )


def load_dataset_with_metadata(data_dir: str, split: str) -> ImageNet100Dataset:
    """Convenient function to load dataset with automatic metadata detection."""
    data_file = os.path.join(data_dir, f"{split}_complete.pt")
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Dataset file not found: {data_file}")
    
    return ImageNet100Dataset(data_file)


# Example usage
if __name__ == "__main__":
    # Initialize converter
    converter = ParquetToTorchConverter(
        image_size=(224, 224),
        chunk_size=1000,
        save_metadata=True
    )
    
    # Get parquet files (using your existing function)
    from main import get_parquet_files
    
    PARQUET_PATH = os.getenv("PARQUET_PATH")
    if PARQUET_PATH:
        # Convert train data
        train_files = get_parquet_files(PARQUET_PATH, 17, "train")
        train_result = converter.convert_and_save(
            train_files, 
            "data_enhanced", 
            "train"
        )
        
        # Convert validation data  
        val_files = get_parquet_files(PARQUET_PATH, 1, "validation")
        val_result = converter.convert_and_save(
            val_files,
            "data_enhanced", 
            "validation"
        )
        
        # Load and test
        train_dataset = load_dataset_with_metadata("data_enhanced", "train")
        print(f"Loaded train dataset with {len(train_dataset)} samples")
        
        # Create DataLoader
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        print(f"DataLoader created with {len(train_loader)} batches")