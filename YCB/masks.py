import os
from PIL import Image

# Define the directories containing images and masks
image_dir = 'YCB/ycb_data/001_chips_can/'
mask_dir = 'YCB/ycb_data/001_chips_can/masks/'

# Get the list of image files in the directory
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

# Iterate over each image file
for image_file in image_files:
    # Load the JPEG image
    image_path = os.path.join(image_dir, image_file)
    image = Image.open(image_path)

    # Construct the corresponding mask file path
    mask_file = image_file.replace('.jpg', '_mask.pbm')
    mask_path = os.path.join(mask_dir, mask_file)

    # Load the mask PBM file
    mask = Image.open(mask_path)

    # Convert the mask to an inverted mask
    mask = Image.eval(mask, lambda x: 255 - x)

    # Find bounding box of the masked area
    bbox = mask.getbbox()

    # Crop the image using the bounding box
    cropped_image = image.crop(bbox)

    # Save the cropped image
    cropped_image.save(os.path.join('YCB/cropped', image_file))
