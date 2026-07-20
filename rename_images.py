# quick_classify.py
import pandas as pd
import os

df = pd.read_csv('data/HiRISE/labels.csv')

# Show first 10 images for classification
for idx, row in df.head(10).iterrows():
    print(f"\n📸 {row['image_name']}")
    print("Current label:", row['label'])
    new_label = input("Enter new label (or press Enter to keep): ")
    if new_label:
        df.at[idx, 'label'] = new_label

df.to_csv('data/HiRISE/labels.csv', index=False)
print("✅ Labels updated!")