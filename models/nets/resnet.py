import torch
import torch.nn as nn
from torchvision import models



# class SplitBatchNorm(nn.BatchNorm2d):
#     def __init__(self, num_features, num_splits, **kw):
#         super().__init__(num_features, **kw)
#         self.num_splits = num_splits

#     def forward(self, input):
#         N, C, H, W = input.shape
#         if self.training or not self.track_running_stats:
#             running_mean_split = self.running_mean.repeat(self.num_splits)
#             running_var_split = self.running_var.repeat(self.num_splits)
#             outcome = nn.functional.batch_norm(
#                 input.view(-1, C * self.num_splits, H, W), running_mean_split, running_var_split,
#                 self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
#                 True, self.momentum, self.eps).view(N, C, H, W)
#             self.running_mean.data.copy_(running_mean_split.view(self.num_splits, C).mean(dim=0))
#             self.running_var.data.copy_(running_var_split.view(self.num_splits, C).mean(dim=0))
#             return outcome
#         else:
#             return nn.functional.batch_norm(
#                 input, self.running_mean, self.running_var,
#                 self.weight, self.bias, False, self.momentum, self.eps)


class ResNet(torch.nn.Module):
    def __init__(self, net_name, num_classes, pretrained=False, is_CIFAR=False):
        super(ResNet, self).__init__()
        model_name_list = sorted(name for name in models.__dict__
                                if name.islower() and not name.startswith("__")
                                and callable(models.__dict__[name]))

        if net_name not in model_name_list:
            assert Exception(f"[!] Networks\' Name is wrong, check net config, \
                               expected: {model_name_list}  \
                               received: {net_name}")
        else:
            resnet_arch = models.__dict__[net_name]

        net = resnet_arch(num_classes=num_classes)
        self.net = []

        ln_in_features = 0
        for name, module in net.named_children():
            if name == 'conv1' and is_CIFAR:
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if isinstance(module, torch.nn.MaxPool2d) and is_CIFAR:
                continue
            if isinstance(module, torch.nn.Linear):
                ln_in_features = module.in_features
                continue
            self.net.append(module)

        self.net = torch.nn.Sequential(*self.net)
        self.fc = torch.nn.Linear(in_features=ln_in_features, out_features=num_classes)

    def forward(self, x, ood_test=False):
        out = self.net(x)
#         out = out.view(-1, self.fc.in_features)
        out = out.view(out.size(0), -1)
        output = self.fc(out)
        if ood_test:
            return output, out
        else:
            return output


class build_ResNet:
#     def __init__(self, arch_name, pretrained=False, bn_momentum=0.01, is_CIFAR=False):
    def __init__(self, arch_name="resnet18", pretrained=False, is_CIFAR=False):
        self.net_name = arch_name
        self.pretrained = pretrained
#         self.bn_momentum = bn_momentum
        self.is_CIFAR = is_CIFAR

    def build(self, num_classes):
        return ResNet(self.net_name,
                      num_classes = num_classes,
                      pretrained=self.pretrained,
                      is_CIFAR=self.is_CIFAR)

if __name__ == '__main__':
    net_builder = build_ResNet("resnet18")
    res18 = net_builder.build(10)
    print(res18)