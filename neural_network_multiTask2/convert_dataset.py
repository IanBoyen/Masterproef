import os
from torchvision.transforms import v2
import torchvision.transforms.functional as F
from PIL import Image

IMG_SIZE = 224
HALF = IMG_SIZE // 2

# Create transformation for images
pre_transform = v2.Compose([
    v2.Lambda(lambda img: F.affine(img, angle=0, translate=(-35, 0), scale=1, shear=0)), # Shift 35 pixels left (Correct center alignment)
    v2.Resize(IMG_SIZE, antialias=True), #Resize Image
    v2.CenterCrop(IMG_SIZE), #Crop Image
])


# Paths
src_root = '../../dataset'
dst_root = '../../dataset2'

# Create dataset2 directory structure
os.makedirs(dst_root, exist_ok=True)

# Loop through each class folder
for folder in os.listdir(src_root):
    folder_src = os.path.join(src_root, folder)
    folder_dst = os.path.join(dst_root, folder)
    os.makedirs(folder_dst, exist_ok=True)

    for filename in os.listdir(folder_src):
        if filename.lower().endswith('.jpg'):
            img_path = os.path.join(folder_src, filename)
            image = Image.open(img_path).convert('RGB')
            image = pre_transform(image)

            base_name = os.path.splitext(filename)[0]

            # TOP
            image_new = image.copy()
            image_part = image_new.crop((0, 0, IMG_SIZE, HALF))
            image_part.save(os.path.join(folder_dst, f"{base_name}_TOP.jpg"))

            # BOTTOM
            image_new = image.copy().rotate(180)
            image_part = image_new.crop((0, 0, IMG_SIZE, HALF))
            image_part.save(os.path.join(folder_dst, f"{base_name}_BOTTOM.jpg"))

            # LEFT
            image_new = image.copy().rotate(-90)
            image_part = image_new.crop((0, 0, IMG_SIZE, HALF))
            image_part.save(os.path.join(folder_dst, f"{base_name}_LEFT.jpg"))

            # RIGHT
            image_new = image.copy().rotate(90)
            image_part = image_new.crop((0, 0, IMG_SIZE, HALF))
            image_part.save(os.path.join(folder_dst, f"{base_name}_RIGHT.jpg"))
