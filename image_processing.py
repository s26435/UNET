import os
import numpy as np
from PIL import Image
import numpy as np
from tqdm import tqdm

def load_images_from_folder(folder, image_size):
    images = []
    files = os.listdir(folder)
    for filename in tqdm(files, desc="Loading images", unit="image"):
        img_path = os.path.join(folder, filename)
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                img = img.resize(image_size)
                images.append(np.array(img))

        except (OSError, Image.DecompressionBombError) as e:
            print(f"Skipping file {filename}: {e}")
    return (np.array(images) / 127.5) - 1
