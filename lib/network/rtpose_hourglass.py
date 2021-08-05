
import paddle.nn as nn
from paddle.nn import initializer as init
import paddle.nn.functional as F

NUM_JOINTS = 18
NUM_LIMBS = 38

class ReLU_(nn.Layer):
    def __init__(self, name=None):
        super(ReLU_, self).__init__()
        self._name = name

    def forward(self, x):
        return F.relu_(x, self._name)

    def extra_repr(self):
        name_str = 'name={}'.format(self._name) if self._name else ''
        return name_str


class Bottleneck(nn.Layer):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.bn1 = nn.BatchNorm2D(inplanes)
        self.conv1 = nn.Conv2D(inplanes, planes, kernel_size=1, bias=True)
        self.bn2 = nn.BatchNorm2D(planes)
        self.conv2 = nn.Conv2D(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn3 = nn.BatchNorm2D(planes)
        self.conv3 = nn.Conv2D(planes, planes * 2, kernel_size=1, bias=True)
        self.relu = ReLU_()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class Hourglass(nn.Layer):
    def __init__(self, block, num_blocks, planes, depth):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.block = block
        self.upsample = nn.Upsample(scale_factor=2)
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth)

    def _make_residual(self, block, num_blocks, planes):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(planes * block.expansion, planes))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, num_blocks, planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res.append(self._make_residual(block, num_blocks, planes))
            if i == 0:
                res.append(self._make_residual(block, num_blocks, planes))
            hg.append(nn.LayerList(res))
        return nn.LayerList(hg)

    def _hour_glass_forward(self, n, x):
        up1 = self.hg[n - 1][0](x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n - 1][1](low1)

        if n > 1:
            low2 = self._hour_glass_forward(n - 1, low1)
        else:
            low2 = self.hg[n - 1][3](low1)
        low3 = self.hg[n - 1][2](low2)
        up2 = self.upsample(low3)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)


class HourglassNet(nn.Layer):
    '''Hourglass model from Newell et al ECCV 2016'''

    def __init__(self, block, num_stacks=2, num_blocks=4, paf_classes=NUM_LIMBS*2, ht_classes=NUM_JOINTS+1):
        super(HourglassNet, self).__init__()

        self.inplanes = 64
        self.num_feats = 128
        self.num_stacks = num_stacks
        self.conv1 = nn.Conv2D(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=True)
        self.bn1 = nn.BatchNorm2D(self.inplanes)
        self.relu = ReLU_()
        self.layer1 = self._make_residual(block, self.inplanes, 1)
        self.layer2 = self._make_residual(block, self.inplanes, 1)
        self.layer3 = self._make_residual(block, self.num_feats, 1)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        # build hourglass modules
        ch = self.num_feats * block.expansion
        hg, res, fc, score_paf, score_ht, fc_, paf_score_, ht_score_ = \
        [], [], [], [], [], [], [], []
        for i in range(num_stacks):
            hg.append(Hourglass(block, num_blocks, self.num_feats, 4))
            res.append(self._make_residual(block, self.num_feats, num_blocks))
            fc.append(self._make_fc(ch, ch))
            score_paf.append(nn.Conv2D(ch, paf_classes, kernel_size=1, bias=True))
            score_ht.append(nn.Conv2D(ch, ht_classes, kernel_size=1, bias=True))            
            if i < num_stacks - 1:
                fc_.append(nn.Conv2D(ch, ch, kernel_size=1, bias=True))
                paf_score_.append(nn.Conv2D(paf_classes, ch,
                                        kernel_size=1, bias=True))
                ht_score_.append(nn.Conv2D(ht_classes, ch,
                                        kernel_size=1, bias=True))                                        
        self.hg = nn.LayerList(hg)
        self.res = nn.LayerList(res)
        self.fc = nn.LayerList(fc)
        self.score_ht = nn.LayerList(score_ht)
        self.score_paf = nn.LayerList(score_paf)        
        self.fc_ = nn.LayerList(fc_)
        self.paf_score_ = nn.LayerList(paf_score_)
        self.ht_score_ = nn.LayerList(ht_score_)
        
        self._initialize_weights_norm()        
        
    def _make_residual(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, outplanes):
        bn = nn.BatchNorm2D(inplanes)
        conv = nn.Conv2D(inplanes, outplanes, kernel_size=1, bias=True)
        return nn.Sequential(
            conv,
            bn,
            self.relu,
        )

    def forward(self, x):
        saved_for_loss = []    
        out = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)

        for i in range(self.num_stacks):
            y = self.hg[i](x)
            y = self.res[i](y)
            y = self.fc[i](y)
            score_paf = self.score_paf[i](y)
            score_ht = self.score_ht[i](y)            
            if i < self.num_stacks - 1:
                fc_ = self.fc_[i](y)
                paf_score_ = self.paf_score_[i](score_paf)
                ht_score_ = self.ht_score_[i](score_ht)                
                x = x + fc_ + paf_score_ + ht_score_
                
        saved_for_loss.append(score_paf)
        saved_for_loss.append(score_ht)

        return (score_paf, score_ht), saved_for_loss

    def _initialize_weights_norm(self):        
       for m in self.modules():
            if isinstance(m, nn.Conv2D):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:  # mobilenet conv2d doesn't add bias
                    init.constant_(m.bias, 0.0) 
            elif isinstance(m, nn.BatchNorm2D):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def hg(**kwargs):
    model = HourglassNet(Bottleneck, num_stacks=kwargs['num_stacks'], 
    num_blocks=kwargs['num_blocks'], paf_classes=kwargs['paf_classes'], 
    ht_classes=kwargs['ht_classes'])
    return model