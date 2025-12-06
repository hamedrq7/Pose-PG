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

import gdown
os.makedirs("./downloads", exist_ok=True)

def gdown_download(url, output_name):
    if os.path.exists(output_name):
        print(f"File already exists: {output_name}")
    else:
        gdown.download(url, output=f'{output_name}', quiet=False)
    
########################################  COCO  ########################################
os.makedirs("./datasets/COCO", exist_ok=True)

# COCO dataset
coco_val_zip = "./downloads/val2017.zip"
coco_val_extract_dir = "./datasets/COCO/val2017"  

coco_train_zip = "./downloads/train2017.zip"
coco_train_extract_dir = "./datasets/COCO/train2017"  

coco_annotation = "./downloads/annotations_trainval2017.zip"
coco_annotation_extract_dir = "./datasets/COCO/annotations"  

download_file("http://images.cocodataset.org/zips/val2017.zip", coco_val_zip)
if not os.path.exists(coco_val_extract_dir):
    run(f"unzip -q -n {coco_val_zip} -d ./datasets/COCO")  # -n: don't overwrite
else:
    print(f"Skipping unzip — found existing directory: {coco_val_extract_dir}")

download_file("http://images.cocodataset.org/zips/train2017.zip", coco_train_zip)
if not os.path.exists(coco_train_extract_dir):
    run(f"unzip -q -n {coco_train_zip} -d ./datasets/COCO")
else:
    print(f"Skipping unzip — found existing directory: {coco_train_extract_dir}")

download_file("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", coco_annotation)
if not os.path.exists(coco_annotation_extract_dir):
    run(f"unzip -q -n {coco_annotation} -d ./datasets/COCO")
else:
    print(f"Skipping unzip — found existing directory: {coco_annotation_extract_dir}")


os.makedirs("./datasets/COCO/person_detection_results", exist_ok=True)
gdown_download('https://drive.google.com/uc?id=1ygw57X-mh0QBfENB-U5DsuSauGIu-8RB', './datasets/COCO/person_detection_results/COCO_val2017_detections_AP_H_56_person.json')

# #######################################  AP10K  ########################################
# # download dataset
# url = "https://drive.google.com/uc?id=1-FNNGcdtAQRehYYkGY1y4wzFNg4iWNad"
# ap10_zip_path = "./downloads/ap-10k.zip"
# gdown_download(url, ap10_zip_path)

# if not os.path.exists("./datasets/ap-10k"):
#     run(f"unzip -q -n {ap10_zip_path} -d ./datasets")
# else:
#     print(f"Skipping unzip — found existing directory: ./datasets/ap-10k")

# #######################################  CrowdPose  ########################################
# # download dataset
# url = "https://drive.google.com/uc?id=1VprytECcLtU4tKP32SYi_7oDRbw7yUTL"
# crowdpose_zip_path = "./downloads/crowdpose.zip"
# crowdpose_extract_dir = './datasets/CrowdPose'
# gdown_download(url, crowdpose_zip_path)

# if not os.path.exists(crowdpose_extract_dir):
#     run(f"unzip -q -n {crowdpose_zip_path} -d {crowdpose_extract_dir}")
# else:
#     print(f"Skipping unzip — found existing directory: {crowdpose_extract_dir}")

# os.makedirs("./datasets/CrowdPose/annotations", exist_ok=True)
# gdown_download('https://drive.google.com/uc?id=18-IwNa6TOGQPE0RqGNjNY1cJOfNC7MXj', './datasets/CrowdPose/annotations/crowdpose_val.json')
# gdown_download('https://drive.google.com/uc?id=13xScmTWqO6Y6m_CjiQ-23ptgX9sC-J9I', './datasets/CrowdPose/annotations/crowdpose_trainval.json')
# gdown_download('https://drive.google.com/uc?id=1b3APtKpc43dx_5FxizbS-EWGvd-zl7Lb', './datasets/CrowdPose/annotations/crowdpose_train.json')
# gdown_download('https://drive.google.com/uc?id=1FUzRj-dPbL1OyBwcIX2BgFPEaY5Yrz7S', './datasets/CrowdPose/annotations/crowdpose_test.json')
# gdown_download('https://drive.google.com/uc?id=13KU2xifSerWCTrJHfbCxoD_BD3zVxiOl', './datasets/CrowdPose/annotations/det_for_crowd_test_0.1_0.5.json')

########################################  ImageNet  ########################################
# import datasets.CustomDS.data_configs.ImageNet_configs as imagenet_configs
# os.makedirs(f"{imagenet_configs.root}", exist_ok=True)

# imagenet_devkit_url = "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz"
# imagenet_val_path = f"{imagenet_configs.root}/ILSVRC2012_devkit_t12.tar.gz"
# download_file(imagenet_devkit_url, imagenet_val_path)

# if not os.path.exists(f'{imagenet_configs.root}/val'):
#     imagenet_val_url = "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar"
#     imagenet_val_path = f"{imagenet_configs.root}/ILSVRC2012_img_val.tar"
#     download_file(imagenet_val_url, imagenet_val_path)
#     results = imagenet_configs.unzip_imagenet_val()
#     print('Unzipped ImageNet Val, status: ', results)
#     if results == True: 
#         print('Deleting zip files')
#         os.remove(imagenet_val_path)
# else:
#     print(f'{imagenet_configs.root}/val exits, skipping download and unzip')

# if not os.path.exists(f'{imagenet_configs.root}/train'):
#     imagenet_train_url = "https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar"
#     imagenet_train_path = f"{imagenet_configs.root}/ILSVRC2012_img_train.tar"
#     download_file(imagenet_train_url, imagenet_train_path)
#     results = imagenet_configs.unzip_imagenet_train()
#     print('Unzipped ImageNet train, status: ', results )
#     if results == True: 
#         print('Deleting zip files')
#         os.remove(imagenet_train_path)
# else:
#     print(f'{imagenet_configs.root}/train exits, skipping download and unzip')


########################################  Model Weights  ########################################
# standard resnet50
download_file(
    "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    "./downloads/standard_resnet50.pth"
)
# madry adv_resnet50 linf 8
dropbox_url = "https://www.dropbox.com/scl/fi/uwr6kbkchhi2t34czbzvh/imagenet_linf_8.pt?rlkey=fxnlz3irzmhvx8cbej7ye3fj5&st=l5msjf1p&dl=1"
download_file(dropbox_url, "./downloads/madry_adversarial_resnet50.pt")

# madry adv_resnet50 linf 4
dropbox_url = "https://www.dropbox.com/scl/fi/u04jwt1ms0pjh3a9luixy/imagenet_linf_4.pt?rlkey=1x79l70m0qx18erxy2yonjkwl&st=8hhaehmj&dl=1"
download_file(dropbox_url, "./downloads/madry_adversarial_resnet50_linf4.pt")

# From here: https://huggingface.co/madrylab/robust-imagenet-models
dropbox_url = "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet50_l2_eps0.05.ckpt"
download_file(dropbox_url, "./downloads/resnet50_l2_eps0.05.ckpt")

# Poseresnet50 pretrained on COCO (256x192), 
# from here https://drive.google.com/drive/folders/1hOTihvbyIxsm5ygDpbUuJ7O_tzv4oXjC 
#           https://github.com/leoxiaobin/deep-high-resolution-net.pytorch?tab=readme-ov-file
url = "https://drive.google.com/uc?id=1G5vwpkN6Dr7k1Y0mpo3Cai0uHklQzNmY"
gdown_download(url, './downloads/pose_resnet_50_256x192.pth')

# Hrnet 48 256x192, trained on COCO 
url = "https://drive.google.com/uc?id=15T2XqPjW7Ex0uyC1miGVYUv7ULOxIyJI"
gdown_download(url, './downloads/pose_hrnet_w48_256x192.pth')

# Poserenset50 pretrained on AP10K (256x256)
download_file("https://download.openmmlab.com/mmpose/animal/resnet/res50_ap10k_256x256-35760eb8_20211029.pth", 
    "./downloads/posersnet50_ap10_256x256.pth")

# Vitpose_small (trained on COCO)
url = "https://drive.google.com/uc?id=1MQlNJO1mPTghyz2uJjI6Z3ui_1CX77E-"
gdown_download(url, './downloads/vitpose_small.pth')
