root = './datasets/IMAGENET'

animal_winds_path = "./datasets/CustomDS/data_configs/imagenet_animal_wnids.json"

def unzip_imagenet_val():
    import torchvision

    ds = torchvision.datasets.ImageNet(f'{root}', split='val',)
    print(len(ds))
    return (len(ds) == 1281167) 

def unzip_imagenet_train():
    import torchvision

    ds = torchvision.datasets.ImageNet(f'{root}', split='train', )
    
    print(len(ds))
    return len(ds) == 50000
 
import json
from torchvision.datasets import ImageNet

class FilteredImageNet(ImageNet):
    """
    ImageNet dataset that only keeps samples whose WNID is in a supplied list.
    """
    def __init__(self, root, split, wnid_file, transform=None, target_transform=None):
        super().__init__(root, split=split,
                         transform=transform,
                         target_transform=target_transform)

        # Load allowed WNIDs
        with open(wnid_file, "r") as f:
            allowed_wnids = set(json.load(f))

        # Map wnid -> original class index
        wnid_to_idx = {wnid: idx for idx, wnid in enumerate(self.wnids)}

        # Compute allowed class indices
        allowed_class_indices = {
            wnid_to_idx[wnid] for wnid in allowed_wnids
        }

        # Build new continuous label mapping
        idx_remap = {old_idx: new_idx
                     for new_idx, old_idx in enumerate(sorted(allowed_class_indices))}

        # Filter samples
        new_samples = []
        for path, label in self.samples:
            if label in allowed_class_indices:
                new_samples.append((path, idx_remap[label]))

        # Replace dataset internals
        self.samples = new_samples
        self.targets = [label for (_, label) in new_samples]
        self.classes = [self.classes[old] for old in sorted(allowed_class_indices)]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        print(f"Filtered ImageNet: {len(new_samples)} samples, "
              f"{len(self.classes)} classes kept.")

if __name__ == "__main__":
    from torchvision import datasets, transforms, models

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    dataset = FilteredImageNet(
        root=root,
        split="val",
        wnid_file=animal_winds_path,
        transform=val_transform
    )

    print(len(dataset))       # number of animal images
    print(len(dataset.classes))   # should be 398
