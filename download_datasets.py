import kagglehub
import os
import shutil

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

print("Downloading IMDB movie reviews dataset...")
# Download IMDB dataset
imdb_path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
print("Path to IMDB dataset files:", imdb_path)

print("\nDownloading Amazon product reviews dataset...")
# Download Amazon product reviews dataset
amazon_path = kagglehub.dataset_download("arhamrumi/amazon-product-reviews")
print("Path to Amazon dataset files:", amazon_path)

# Copy files to our data directory for easier access
print("\nCopying files to data directory...")
if os.path.exists(imdb_path):
    for file in os.listdir(imdb_path):
        src = os.path.join(imdb_path, file)
        dst = os.path.join("data", f"imdb_{file}")
        shutil.copy2(src, dst)
        print(f"Copied: {file} -> data/imdb_{file}")

if os.path.exists(amazon_path):
    for file in os.listdir(amazon_path):
        src = os.path.join(amazon_path, file)
        dst = os.path.join("data", f"amazon_{file}")
        shutil.copy2(src, dst)
        print(f"Copied: {file} -> data/amazon_{file}")

print("\nDataset download complete!")
print("Files in data directory:")
for file in os.listdir("data"):
    print(f"  - {file}")
