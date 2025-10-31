import os
import subprocess
import urllib.request

def run(cmd: str):
    """Run a shell command safely with output."""
    print(f"\nRunning: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def download_file(url: str, destination: str):
    """Download a file if it doesn't already exist."""
    if os.path.exists(destination):
        print(f"File already exists: {destination}")
        return
    print(f"Downloading {url} -> {destination}")
    urllib.request.urlretrieve(url, destination)
    print(f"Download complete: {destination}")

# Install dependencies
if os.path.exists("requirements.txt"):
    print("\nInstalling dependencies from requirements.txt...")
    run("pip install -r requirements.txt")
else:
    print("\nNo requirements.txt found — skipping dependency installation.")

print("\nAll downloads and setup complete.")

# # Create directories
os.makedirs("./downloads", exist_ok=True)
os.makedirs("./datasets/COCO", exist_ok=True)

# COCO dataset
coco_val_zip = "./downloads/val2017.zip"
coco_val_extract_dir = "./datasets/COCO/val2017"  

# coco_train_zip = "./downloads/train2017.zip"
# coco_train_extract_dir = "./datasets/COCO/train2017"  

coco_annotation = "./downloads/annotations_trainval2017.zip.zip"
coco_annotation_extract_dir = "./datasets/COCO/annotations"  

download_file("http://images.cocodataset.org/zips/val2017.zip", coco_val_zip)
if not os.path.exists(coco_val_extract_dir):
    run(f"unzip -q -n {coco_val_zip} -d ./datasets/COCO")  # -n: don't overwrite
else:
    print(f"Skipping unzip — found existing directory: {coco_val_extract_dir}")

# download_file("http://images.cocodataset.org/zips/train2017.zip", coco_train_zip)
# if not os.path.exists(coco_train_extract_dir):
#     run(f"unzip -q -n {coco_train_zip} -d ./datasets/COCO")
# else:
#     print(f"Skipping unzip — found existing directory: {coco_train_extract_dir}")

download_file("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", coco_annotation)
if not os.path.exists(coco_annotation_extract_dir):
    run(f"unzip -q -n {coco_annotation} -d ./datasets/COCO")
else:
    print(f"Skipping unzip — found existing directory: {coco_annotation_extract_dir}")


# Model weights
download_file(
    "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    "./downloads/standard_resnet50.pth"
)

# Dropbox link with direct download enabled
dropbox_url = "https://www.dropbox.com/scl/fi/uwr6kbkchhi2t34czbzvh/imagenet_linf_8.pt?rlkey=fxnlz3irzmhvx8cbej7ye3fj5&st=l5msjf1p&dl=1"
download_file(dropbox_url, "./downloads/madry_adversarial_resnet50.pt")

# Poseresnet50 pretrained on COCO (256x192)
import gdown

url = "https://drive.google.com/uc?id=1G5vwpkN6Dr7k1Y0mpo3Cai0uHklQzNmY"
gdown.download(url, output='./downloads/pose_resnet_50_256x192.pth', quiet=False)

# Poserenset50 pretrained on AP10K (256x256)
download_file("https://download.openmmlab.com/mmpose/animal/resnet/res50_ap10k_256x256-35760eb8_20211029.pth", 
    "./downloads/posersnet50_ap10_256x256.pth")
