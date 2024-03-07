import os
import json
import urllib.request
import shutil
import tarfile

output_directory = "YCB/ycb_data"  # Save data to a subdirectory called 'ycb_data'

# Fetch object names
def fetch_objects(url):
    with urllib.request.urlopen(url) as response:
        html = response.read().decode('utf-8')
        objects = json.loads(html)
        return objects["objects"]

# Download file
def download_file(url, filename):
    with urllib.request.urlopen(url) as response, open(filename, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)

# Extract .tgz file
def extract_tgz(filename, output_directory):
    with tarfile.open(filename, 'r:gz') as tar:
        tar.extractall(output_directory)

# URL for fetching object names
objects_url = "http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/objects.json"

# Fetch object names
object_names = fetch_objects(objects_url)

# Download berkeley_rgb_highres data for all objects
for object_name in object_names:
    url = f"http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/berkeley/{object_name}/{object_name}_berkeley_rgb_highres.tgz"
    filename = f"{output_directory}/{object_name}_berkeley_rgb_highres.tgz"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    download_file(url, filename)
    print(f"Downloaded berkeley_rgb_highres data for {object_name}")
    extract_tgz(filename, output_directory)
    print(f"Extracted berkeley_rgb_highres data for {object_name}")
