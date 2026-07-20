# create_mock_data.py
import os
import pandas as pd
from PIL import Image
import numpy as np

def create_mock_data(num_samples=50):
    """
    Create mock images and labels for testing.
    """
    print("🔄 Creating mock data...")
    
    # Create directories
    os.makedirs('data/HiRISE/images', exist_ok=True)
    os.makedirs('models/trained_models', exist_ok=True)
    
    # Define classes
    classes = ['bright dune', 'crater', 'dark dune', 'impact ejecta', 
               'slope streak', 'spider', 'swiss cheese']
    
    # Create images and labels
    data = []
    for i in range(num_samples):
        # Create a random colored image
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img_name = f'mock_img_{i:03d}.jpg'
        img_path = os.path.join('data/HiRISE/images', img_name)
        Image.fromarray(img).save(img_path)
        
        # Assign random labels
        label = np.random.choice(classes)
        data.append({'image_name': img_name, 'label': label})
    
    # Save to CSV
    df = pd.DataFrame(data)
    df.to_csv('data/HiRISE/labels.csv', index=False)
    
    print(f"✅ Created {len(df)} mock images with labels")
    print("\n📊 Class Distribution:")
    class_counts = df['label'].value_counts()
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} samples")
    
    return df

if __name__ == "__main__":
    create_mock_data(50)
    print("\n✅ Data created successfully!")
    print("📁 Images: data/HiRISE/images/")
    print("📄 Labels: data/HiRISE/labels.csv")
    print("\n🚀 Now run: python train.py")