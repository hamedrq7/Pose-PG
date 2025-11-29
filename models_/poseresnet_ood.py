import torch
from torch import nn

import os 
import sys 
# relative import hacks (sorry)
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)  # for bash user
os.chdir(parentdir)  # for pycharm user


from models_.modules import BasicBlock, Bottleneck


resnet_spec = {
    18: (BasicBlock, [2, 2, 2, 2]),
    34: (BasicBlock, [3, 4, 6, 3]),
    50: (Bottleneck, [3, 4, 6, 3]),
    101: (Bottleneck, [3, 4, 23, 3]),
    152: (Bottleneck, [3, 8, 36, 3])
}


# derived from https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
class PoseResNetOOD(nn.Module):
    def __init__(self, resnet_size=50, nof_joints=17, bn_momentum=0.1):
        super(PoseResNetOOD, self).__init__()

        assert resnet_size in resnet_spec.keys()
        block, layers = resnet_spec[resnet_size]

        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], bn_momentum=bn_momentum)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, bn_momentum=bn_momentum)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, bn_momentum=bn_momentum)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, bn_momentum=bn_momentum)

        # used for deconv layers
        self.deconv_with_bias = False
        self.deconv_layers = self._make_deconv_layer(
            3,
            [256, 256, 256],
            [4, 4, 4],
            bn_momentum=bn_momentum
        )

        self.ood_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 1)
        )

        self.final_layer = nn.Conv2d(
            in_channels=256,
            out_channels=nof_joints,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def get_joint_prototypes(self):
        return self.final_layer
    
    def _make_layer(self, block, planes, blocks, stride=1, bn_momentum=0.1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_momentum),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels, bn_momentum=0.1):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=bn_momentum))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x, return_feats=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        feats = self.deconv_layers(x)
        pose_out = self.final_layer(feats)
        ood_out = self.ood_head(feats)

        if return_feats:
            return feats, ood_out, pose_out
        return ood_out, pose_out




if __name__ == '__main__':

    model = PoseResNetOOD(50, 17, 0.1)
    # print(model)
    # model.load_state_dict(
    #     torch.load('./weights/pose_resnet_50_256x192.pth')
    # )
    # print('ok!!')

    if torch.cuda.is_available() and False:
        torch.backends.cudnn.deterministic = True
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    print(device)
    print(model)
    model = model.to(device)
    print(model)
    from torchinfo import summary
    summary(model, (1, 3, 256, 192))
    exit()

    pretrained_weight_path = 'C:/Users/hamed/Downloads/imagenet_linf_8.pt'
    
    #### From torchvission, standard r50
    # from torchvision.models import resnet50, ResNet50_Weights
    # # Using pretrained weights:
    # # resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    # # resnet50(weights="IMAGENET1K_V1")

    # from torchinfo import summary
    # summary(resnet, (1, 3, 256, 192))
    from training.Train import load_pretrained
    model = load_pretrained(pretrained_weight_path, device)
