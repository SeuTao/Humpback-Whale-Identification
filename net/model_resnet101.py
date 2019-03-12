from imagenet_pretrain_model.senet import *
# from utils import *
from MagrinLinear import *
BatchNorm2d = nn.BatchNorm2d

import torchvision.models as tvm
###########################################################################################3
class BinaryHead(nn.Module):

    def __init__(self, num_class=10008, emb_size = 2048, s = 16.0):
        super(BinaryHead,self).__init__()
        self.s = s
        self.fc = nn.Sequential(nn.Linear(emb_size, num_class))

    def forward(self, fea):
        fea = l2_norm(fea)
        logit = self.fc(fea)*self.s
        return logit

###########################################################################################3
class MarginHead(nn.Module):

    def __init__(self, num_class=10008, emb_size = 2048, s=64., m=0.5):
        super(MarginHead,self).__init__()
        self.fc = MagrginLinear(embedding_size=emb_size, classnum=num_class , s=s, m=m)

    def forward(self, fea, label, is_infer):
        fea = l2_norm(fea)
        logit = self.fc(fea, label, is_infer)
        return logit

###########################################################################################3
class Net(nn.Module):

    def load_pretrain(self, pretrain_file):
        pretrain_state_dict = torch.load(pretrain_file)
        state_dict = self.state_dict()
        keys = list(state_dict.keys())
        for key in keys:
            state_dict[key] = pretrain_state_dict[r'module.'+key]
            print(key)
        self.load_state_dict(state_dict)
        print('')

    def __init__(self, num_class=10008, s1 = 64 , m1 = 0.5, s2 = 64):
        super(Net,self).__init__()

        self.s1 = s1
        self.m1 = m1
        self.s2 = s2

        self.basemodel = tvm.resnet101(pretrained=True)
        self.basemodel.avgpool = nn.AdaptiveAvgPool2d(1)
        self.basemodel.last_linear = nn.Sequential()
        self.basemodel.layer0 = nn.Sequential(self.basemodel.conv1,
                                              self.basemodel.bn1,
                                              self.basemodel.relu,
                                              self.basemodel.maxpool)

        emb_size = 2048
        self.fea_bn = nn.BatchNorm1d(emb_size)
        self.fea_bn.bias.requires_grad_(False)

        self.margin_head = MarginHead(num_class, emb_size=emb_size, s = self.s1, m = self.m1)
        self.binary_head = BinaryHead(num_class, emb_size=emb_size, s = self.s2)

    def forward(self, x, label = None, is_infer = None):
        mean = [0.485, 0.456, 0.406]  # rgb
        std = [0.229, 0.224, 0.225]

        x = torch.cat([
            (x[:, [0]] - mean[0]) / std[0],
            (x[:, [1]] - mean[1]) / std[1],
            (x[:, [2]] - mean[2]) / std[2],
        ], 1)

        x = self.basemodel.layer0(x)
        x = self.basemodel.layer1(x)
        x = self.basemodel.layer2(x)
        x = self.basemodel.layer3(x)
        x = self.basemodel.layer4(x)

        x = F.adaptive_avg_pool2d(x,1)
        fea = x.view(x.size(0), -1)
        fea = self.fea_bn(fea)
        logit_binary = self.binary_head(fea)
        logit_margin = self.margin_head(fea, label = label, is_infer = is_infer)

        return logit_binary, logit_margin, fea
