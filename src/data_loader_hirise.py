# src/data_loader_hirise.py - Fixed version with better image search
# src/data_loader_hirise.py - Add this at the top after imports
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import glob
import numpy as np
import re
import torch  # <-- Add this line

# Define DEVICE globally
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HiRISEDatasetV3(Dataset):
    def __init__(self, image_dir, dataframe, transform=None):
        """
        Specialized dataset for HiRISE V3 data.
        Args:
            image_dir (str): Path to map-proj-v3/ directory
            dataframe (DataFrame): With columns 'image_name' and 'label'
            transform (callable, optional): Image transformations
        """
        self.image_dir = image_dir
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        
        # Pre-build a cache of available images for faster lookup
        self._build_image_cache()

    def _build_image_cache(self):
        """Build a cache of available images in the directory."""
        self.image_cache = {}
        if os.path.exists(self.image_dir):
            # Get all files in the directory
            all_files = glob.glob(os.path.join(self.image_dir, '*.*'))
            for file_path in all_files:
                file_name = os.path.basename(file_path)
                # Store with and without extension
                base_name = os.path.splitext(file_name)[0]
                self.image_cache[file_name] = file_path
                self.image_cache[base_name] = file_path
                # Also store lowercase versions
                self.image_cache[file_name.lower()] = file_path
                self.image_cache[base_name.lower()] = file_path
        print(f"✅ Built image cache with {len(self.image_cache)} entries")

    def _find_image_path(self, img_name):
        """Find image path using cache."""
        # Try exact match first
        if img_name in self.image_cache:
            return self.image_cache[img_name]
        
        # Try lowercase
        if img_name.lower() in self.image_cache:
            return self.image_cache[img_name.lower()]
        
        # Try without extension
        base_name = os.path.splitext(img_name)[0]
        if base_name in self.image_cache:
            return self.image_cache[base_name]
        
        # Try lowercase without extension
        if base_name.lower() in self.image_cache:
            return self.image_cache[base_name.lower()]
        
        # Try to find by partial match (for rotated/flipped images)
        # Remove common suffixes like -r90, -r180, -r270, -flip, etc.
        clean_name = re.sub(r'-(r90|r180|r270|flip|vflip|hflip|bright|dark)\d*$', '', base_name)
        clean_name = re.sub(r'-(rot|rotated|flip|flipped)\d*$', '', clean_name)
        
        if clean_name != base_name:
            # Search for any file starting with clean_name
            for cached_name in self.image_cache:
                if cached_name.startswith(clean_name) or cached_name.startswith(clean_name.lower()):
                    return self.image_cache[cached_name]
        
        # Try glob search as last resort
        pattern = os.path.join(self.image_dir, base_name + '.*')
        files = glob.glob(pattern)
        if files:
            return files[0]
        
        # Try with partial match
        pattern = os.path.join(self.image_dir, '*' + base_name + '*.*')
        files = glob.glob(pattern)
        if files:
            return files[0]
        
        return None

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Get image name and label
        img_name = str(self.dataframe.iloc[idx]['image_name'])
        label = self.dataframe.iloc[idx]['label']
        
        # Find image path using cache
        img_path = self._find_image_path(img_name)
        
        if img_path is None:
            # If still not found, create dummy image
            print(f"⚠️ Image not found: {img_name}")
            image = Image.fromarray(np.random.randint(0, 255, (227, 227, 3), dtype=np.uint8))
        else:
            try:
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                image = Image.fromarray(np.random.randint(0, 255, (227, 227, 3), dtype=np.uint8))
        
        if self.transform:
            image = self.transform(image)

        return image, label


def load_hirise_data(data_dir="data/HiRISE/map-proj-v3/", 
                     labels_file="data/HiRISE/labels-map-proj-v3.txt",
                     classmap_file="data/HiRISE/landmarks_map-proj-v3_classmap.csv"):
    """
    Load HiRISE V3 dataset with proper labels.
    """
    print("📂 Loading HiRISE V3 dataset...")
    
    # Check if directories exist
    if not os.path.exists(data_dir):
        print(f"❌ Directory not found: {data_dir}")
        print("Please extract the HiRISE dataset to this location.")
        return None, None, None, None
    
    # Load class mapping with flexible column names
    class_map = {}
    if os.path.exists(classmap_file):
        class_df = pd.read_csv(classmap_file)
        print(f"✅ Loaded classmap file with columns: {class_df.columns.tolist()}")
        
        # Try to find the right columns
        id_col = None
        name_col = None
        
        # Look for ID column
        for col in class_df.columns:
            if col.lower() in ['id', 'class_id', 'classid', 'label_id', 'labelid']:
                id_col = col
                break
        # If not found, use first column
        if id_col is None:
            id_col = class_df.columns[0]
            print(f"ℹ️ Using '{id_col}' as ID column")
        
        # Look for Name column
        for col in class_df.columns:
            if col.lower() in ['name', 'class_name', 'classname', 'label_name', 'labelname', 'semantic']:
                name_col = col
                break
        # If not found, use second column
        if name_col is None and len(class_df.columns) > 1:
            name_col = class_df.columns[1]
            print(f"ℹ️ Using '{name_col}' as Name column")
        elif name_col is None:
            name_col = id_col
            print("⚠️ No name column found, using IDs as names")
        
        # Build class map
        for _, row in class_df.iterrows():
            class_id = row[id_col]
            class_name = row[name_col] if name_col != id_col else str(class_id)
            class_map[class_id] = class_name
        
        print(f"✅ Loaded {len(class_map)} class mappings")
    else:
        print(f"⚠️ Classmap file not found: {classmap_file}")
        print("Will use class IDs directly.")
    
    # Load labels
    labels = []
    if os.path.exists(labels_file):
        with open(labels_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_name = parts[0]
                    class_id = int(parts[1])
                    labels.append({'image_name': img_name, 'class_id': class_id})
        print(f"✅ Loaded {len(labels)} image labels")
    else:
        print(f"❌ Labels file not found: {labels_file}")
        return None, None, None, None
    
    # Create DataFrame
    df = pd.DataFrame(labels)
    
    # Add semantic class names
    if class_map:
        df['label'] = df['class_id'].map(class_map)
        # Check for unmapped classes
        unmapped = df[df['label'].isna()]
        if len(unmapped) > 0:
            print(f"⚠️ {len(unmapped)} images have unmapped class IDs")
            # Show unmapped classes
            unmapped_classes = df[df['label'].isna()]['class_id'].unique()
            print(f"   Unmapped class IDs: {unmapped_classes[:10]}")
            # Fill with 'unknown' for unmapped
            df['label'] = df['label'].fillna('unknown')
    else:
        # Use class IDs as labels
        df['label'] = df['class_id'].astype(str)
        print("ℹ️ Using class IDs as labels (no class mapping available)")
    
    # Print class distribution
    print("\n📊 Class Distribution (top 20):")
    class_counts = df['label'].value_counts()
    for class_name, count in class_counts.head(20).items():
        print(f"  {class_name}: {count} samples")
    if len(class_counts) > 20:
        print(f"  ... and {len(class_counts) - 20} more classes")
    
    # Encode labels
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['label'])
    num_classes = len(le.classes_)
    print(f"\n✅ Total classes: {num_classes}")
    
    return df, le, class_map, num_classes


def get_hirise_data_loaders(data_dir="data/HiRISE/map-proj-v3/",
                            labels_file="data/HiRISE/labels-map-proj-v3.txt",
                            classmap_file="data/HiRISE/landmarks_map-proj-v3_classmap.csv",
                            batch_size=64, test_size=0.2, val_size=0.1):
    """
    Create DataLoaders for HiRISE V3 dataset.
    """
    # Load data
    df, le, class_map, num_classes = load_hirise_data(data_dir, labels_file, classmap_file)
    
    if df is None:
        return None, None, None, None, None
    
    # Check if we have enough samples for splitting
    if len(df) < 10:
        print("⚠️ Very few samples. Using all data for training.")
        train_df = df
        val_df = df
        test_df = df
    else:
        # Split data
        try:
            train_val_df, test_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df['label_encoded'])
            train_df, val_df = train_test_split(train_val_df, test_size=val_size/(1-test_size), 
                                                random_state=42, stratify=train_val_df['label_encoded'])
        except ValueError as e:
            print(f"⚠️ Stratified split failed: {e}")
            print("Using simple random split instead...")
            train_val_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
            train_df, val_df = train_test_split(train_val_df, test_size=val_size/(1-test_size), random_state=42)
    
    print(f"\n📊 Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Transforms (HiRISE images are 227x227)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match ResNet input
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = HiRISEDatasetV3(data_dir, train_df[['image_name', 'label_encoded']].rename(
        columns={'label_encoded': 'label'}), transform=train_transform)
    val_dataset = HiRISEDatasetV3(data_dir, val_df[['image_name', 'label_encoded']].rename(
        columns={'label_encoded': 'label'}), transform=val_test_transform)
    test_dataset = HiRISEDatasetV3(data_dir, test_df[['image_name', 'label_encoded']].rename(
        columns={'label_encoded': 'label'}), transform=val_test_transform)
    
    # Create DataLoaders (reduce num_workers for Windows)
    num_workers = 0 if os.name == 'nt' else 4
    
    train_loader = DataLoader(train_dataset, batch_size=min(batch_size, len(train_dataset)), 
                              shuffle=True, num_workers=num_workers, pin_memory=(DEVICE != 'cpu'))
    val_loader = DataLoader(val_dataset, batch_size=min(batch_size, len(val_dataset)), 
                            shuffle=False, num_workers=num_workers, pin_memory=(DEVICE != 'cpu'))
    test_loader = DataLoader(test_dataset, batch_size=min(batch_size, len(test_dataset)), 
                             shuffle=False, num_workers=num_workers, pin_memory=(DEVICE != 'cpu'))
    
    return train_loader, val_loader, test_loader, le, num_classes


# Quick test function
if __name__ == "__main__":
    DEVICE = 'cpu'  # Default for testing
    
    print("Testing HiRISE data loader...")
    train_loader, val_loader, test_loader, le, num_classes = get_hirise_data_loaders()
    
    if train_loader is not None:
        print(f"\n✅ Success! Loaded {num_classes} classes")
        print(f"📊 Train batches: {len(train_loader)}")
        print(f"📊 Val batches: {len(val_loader)}")
        print(f"📊 Test batches: {len(test_loader)}")
        
        # Test one batch
        print("\n📸 Testing one batch...")
        for images, labels in train_loader:
            print(f"📸 Batch shape: {images.shape}")
            print(f"🏷️ Labels shape: {labels.shape}")
            print(f"🏷️ First 10 labels: {labels[:10]}")
            break
    else:
        print("❌ Failed to load data")