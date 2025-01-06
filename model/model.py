# Source: https://github.com/fungtion/DANN/tree/476147f70bb818a63bb3461a6ecc12f97f7ab15e

import torch.nn as nn
from torch.autograd import Function


class CNNModel(nn.Module):

    def __init__(self, opt):
        super(CNNModel, self).__init__()
        self.w, self.h = opt.image_size, opt.image_size
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(3, 16, kernel_size=3, padding = 1))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(16))
        self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_conv2', nn.Conv2d(16, 32, kernel_size=3, padding = 1))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(32))
        self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature.add_module('f_relu2', nn.ReLU(True))
        self.feature.add_module('f_conv3', nn.Conv2d(32, 64, kernel_size=3, padding = 1))
        self.feature.add_module('f_bn3', nn.BatchNorm2d(64))
        self.feature.add_module('f_pool3', nn.MaxPool2d(2))
        self.feature.add_module('f_relu3', nn.ReLU(True))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(64 * (self.w//8) * (self.w//8), 100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, opt.n_classes))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax())

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(64 * (self.w//8) * (self.w//8), 100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha, DOMAIN_ADAPTATION_HEAD = True):
        input_data = input_data.expand(input_data.data.shape[0], 3, input_data.size(2), input_data.size(3))
        feature = self.feature(input_data)
        feature = feature.view(-1, 64 * (self.w//8) * (self.w//8))
        class_output = self.class_classifier(feature)
        if DOMAIN_ADAPTATION_HEAD:
            reverse_feature = ReverseLayerF.apply(feature, alpha)
            domain_output = self.domain_classifier(reverse_feature)
        else:
            domain_output = None

        return class_output, domain_output

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

