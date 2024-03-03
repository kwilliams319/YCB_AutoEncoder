import os
from PIL import Image

# Define the directory containing the JPEG images
image_dir = 'YCB/cropped/'

# Create a directory to store padded images if it doesn't exist
output_dir = 'YCB/padded'
os.makedirs(output_dir, exist_ok=True)

# Get the list of JPEG image files in the directory
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

# Initialize variables to store maximum dimensions
max_width, max_height = 0, 0

# Find the maximum dimensions
for image_file in image_files:
    # Load the JPEG image
    image_path = os.path.join(image_dir, image_file)
    image = Image.open(image_path)

    # Update maximum dimensions if necessary
    max_width = max(max_width, image.width)
    max_height = max(max_height, image.height)

# Iterate over each image file
for image_file in image_files:
    # Load the JPEG image
    image_path = os.path.join(image_dir, image_file)
    image = Image.open(image_path)

    # Create a new blank image with the maximum dimensions
    padded_image = Image.new('RGB', (max_width, max_height), color='black')

    # Calculate the position to paste the original image
    left = (max_width - image.width) // 2
    top = (max_height - image.height) // 2

    # Paste the original image onto the padded image
    padded_image.paste(image, (left, top))

    # Save the padded image to the output directory
    padded_image.save(os.path.join(output_dir, image_file))
