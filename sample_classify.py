# sample_classify.py - Classify a sample, then use AI for the rest
import pandas as pd
import os
import random

def sample_classify():
    # Load labels
    df = pd.read_csv('data/HiRISE/labels.csv')
    
    # Select random sample of 50 images
    sample_size = min(50, len(df))
    sample_indices = random.sample(range(len(df)), sample_size)
    
    terrain_types = {
        '1': 'rocky', '2': 'sandy', '3': 'crater', '4': 'smooth',
        '5': 'bright dune', '6': 'dark dune', '7': 'impact ejecta',
        '8': 'slope streak', '9': 'spider', '10': 'swiss cheese'
    }
    
    print("📝 Terrain types:")
    for key, value in terrain_types.items():
        print(f"  {key}. {value}")
    print("-" * 60)
    
    classified = 0
    for idx in sample_indices:
        row = df.iloc[idx]
        print(f"\n📸 {row['image_name']}")
        print(f"   Current: {row['label']}")
        
        choice = input("   Enter terrain number (1-10) or 's' to skip: ").strip()
        
        if choice in terrain_types:
            df.at[df.index[idx], 'label'] = terrain_types[choice]
            classified += 1
            print(f"   ✅ Labeled as: {terrain_types[choice]}")
        elif choice.lower() == 's':
            print("   ⏭️ Skipped")
        else:
            print("   ❌ Invalid. Skipping.")
    
    # Save
    df.to_csv('data/HiRISE/labels.csv', index=False)
    print(f"\n✅ Classified {classified} images out of {sample_size}")
    
    # Show summary
    print("\n📊 Updated Class Distribution:")
    class_counts = df['label'].value_counts()
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} samples")

if __name__ == "__main__":
    sample_classify()