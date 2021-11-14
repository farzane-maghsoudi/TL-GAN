import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torchvision import models
from block import fusions
from collections import OrderedDict 

class adaILN(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9, using_moving_average=True, using_bn=False):
        super(adaILN, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.num_features = num_features
    
        if self.using_bn:
            self.rho = Parameter(torch.Tensor(1, num_features, 3))
            self.rho[:,:,0].data.fill_(3)
            self.rho[:,:,1].data.fill_(1)
            self.rho[:,:,2].data.fill_(1)
            self.register_buffer('running_mean', torch.zeros(1, num_features, 1,1))
            self.register_buffer('running_var', torch.zeros(1, num_features, 1,1))
            self.running_mean.zero_()
            self.running_var.zero_()
        else:
            self.rho = Parameter(torch.Tensor(1, num_features, 2))
            self.rho[:,:,0].data.fill_(3.2)
            self.rho[:,:,1].data.fill_(1)

    def forward(self, input, gamma, beta):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        softmax = nn.Softmax(2)
        rho = softmax(self.rho)
        
        
        if self.using_bn:
            if self.training:
                bn_mean, bn_var = torch.mean(input, dim=[0, 2, 3], keepdim=True), torch.var(input, dim=[0, 2, 3], keepdim=True)
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum)
                    self.running_mean.add_((1 - self.momentum) * bn_mean.data)
                    self.running_var.mul_(self.momentum)
                    self.running_var.add_((1 - self.momentum) * bn_var.data)
                else:
                    self.running_mean.add_(bn_mean.data)
                    self.running_var.add_(bn_mean.data ** 2 + bn_var.data)
            else:
                bn_mean = torch.autograd.Variable(self.running_mean)
                bn_var = torch.autograd.Variable(self.running_var)
            out_bn = (input - bn_mean) / torch.sqrt(bn_var + self.eps)
            rho_0 = rho[:,:,0]
            rho_1 = rho[:,:,1]
            rho_2 = rho[:,:,2]

            rho_0 = rho_0.view(1, self.num_features, 1,1)
            rho_1 = rho_1.view(1, self.num_features, 1,1)
            rho_2 = rho_2.view(1, self.num_features, 1,1)
            rho_0 = rho_0.expand(input.shape[0], -1, -1, -1)
            rho_1 = rho_1.expand(input.shape[0], -1, -1, -1)
            rho_2 = rho_2.expand(input.shape[0], -1, -1, -1)
            out = rho_0 * out_in + rho_1 * out_ln + rho_2 * out_bn
        else:
            rho_0 = rho[:,:,0]
            rho_1 = rho[:,:,1]
            rho_0 = rho_0.view(1, self.num_features, 1,1)
            rho_1 = rho_1.view(1, self.num_features, 1,1)
            rho_0 = rho_0.expand(input.shape[0], -1, -1, -1)
            rho_1 = rho_1.expand(input.shape[0], -1, -1, -1)
            out = rho_0 * out_in + rho_1 * out_ln

        out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)
        return out


class ILN(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9, using_moving_average=True, using_bn=False):
        super(ILN, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.num_features = num_features
    
        if self.using_bn:
            self.rho = Parameter(torch.Tensor(1, num_features, 3))
            self.rho[:,:,0].data.fill_(1)
            self.rho[:,:,1].data.fill_(3)
            self.rho[:,:,2].data.fill_(3)
            self.register_buffer('running_mean', torch.zeros(1, num_features, 1,1))
            self.register_buffer('running_var', torch.zeros(1, num_features, 1,1))
            self.running_mean.zero_()
            self.running_var.zero_()
        else:
            self.rho = Parameter(torch.Tensor(1, num_features, 2))
            self.rho[:,:,0].data.fill_(1)
            self.rho[:,:,1].data.fill_(3.2)

        self.gamma = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.gamma.data.fill_(1.0)
        self.beta.data.fill_(0.0)

    def forward(self, input):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        
        softmax = nn.Softmax(2)
        rho = softmax(self.rho)
        
        if self.using_bn:
            if self.training:
                bn_mean, bn_var = torch.mean(input, dim=[0, 2, 3], keepdim=True), torch.var(input, dim=[0, 2, 3], keepdim=True)
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum)
                    self.running_mean.add_((1 - self.momentum) * bn_mean.data)
                    self.running_var.mul_(self.momentum)
                    self.running_var.add_((1 - self.momentum) * bn_var.data)
                else:
                    self.running_mean.add_(bn_mean.data)
                    self.running_var.add_(bn_mean.data ** 2 + bn_var.data)
            else:
                bn_mean = torch.autograd.Variable(self.running_mean)
                bn_var = torch.autograd.Variable(self.running_var)
            out_bn = (input - bn_mean) / torch.sqrt(bn_var + self.eps)
            rho_0 = rho[:,:,0]
            rho_1 = rho[:,:,1]
            rho_2 = rho[:,:,2]

            rho_0 = rho_0.view(1, self.num_features, 1,1)
            rho_1 = rho_1.view(1, self.num_features, 1,1)
            rho_2 = rho_2.view(1, self.num_features, 1,1)
            rho_0 = rho_0.expand(input.shape[0], -1, -1, -1)
            rho_1 = rho_1.expand(input.shape[0], -1, -1, -1)
            rho_2 = rho_2.expand(input.shape[0], -1, -1, -1)
            out = rho_0 * out_in + rho_1 * out_ln + rho_2 * out_bn
        else:
            rho_0 = rho[:,:,0]
            rho_1 = rho[:,:,1]
            rho_0 = rho_0.view(1, self.num_features, 1,1)
            rho_1 = rho_1.view(1, self.num_features, 1,1)
            rho_0 = rho_0.expand(input.shape[0], -1, -1, -1)
            rho_1 = rho_1.expand(input.shape[0], -1, -1, -1)
            out = rho_0 * out_in + rho_1 * out_ln
        
        out = out * self.gamma.expand(input.shape[0], -1, -1, -1) + self.beta.expand(input.shape[0], -1, -1, -1)
        return out


class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=7):
        super(Discriminator, self).__init__()
		
        en1_1 = [nn.ReflectionPad2d(4), nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True), nn.utils.spectral_norm(nn.Conv2d(256, 64, 1, bias=True)), nn.LeakyReLU(0.2, True)]		
        en2_1 = [nn.ReflectionPad2d(2), nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True), nn.utils.spectral_norm(nn.Conv2d(512, 64, 1, bias=True)), nn.LeakyReLU(0.2, True)]
        en3_1 = [nn.ReflectionPad2d(1), nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True), nn.utils.spectral_norm(nn.Conv2d(1024, 64, 1, bias=True)), nn.LeakyReLU(0.2, True)]
        aff1_1 = [nn.Conv2d(64, 1, 1, bias=True)]
        aff2_1 = [nn.Conv2d(64, 1, 1, bias=True)]
        aff3_1 = [nn.Conv2d(64, 1, 1, bias=True)]
        
        en1_2 = [nn.ReflectionPad2d(4), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.utils.spectral_norm(nn.Conv2d(256, 128, 1, bias=True)), nn.LeakyReLU(0.2, True)]		
        en2_2 = [nn.ReflectionPad2d(2), nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True), nn.utils.spectral_norm(nn.Conv2d(512, 128, 1, bias=True)), nn.LeakyReLU(0.2, True)]
        en3_2 = [nn.ReflectionPad2d(1), nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True), nn.utils.spectral_norm(nn.Conv2d(1024, 128, 1, bias=True)), nn.LeakyReLU(0.2, True)]
        aff1_2 = [nn.Conv2d(128, 1, 1, bias=True)]
        aff2_2 = [nn.Conv2d(128, 1, 1, bias=True)]
        aff3_2 = [nn.Conv2d(128, 1, 1, bias=True)]
        
        en1_3 = [nn.ReflectionPad2d(4), nn.Upsample(scale_factor=1, mode='bilinear', align_corners=True), nn.utils.spectral_norm(nn.Conv2d(256, 256, 1, bias=True)), nn.LeakyReLU(0.2, True)]		
        en2_3 = [nn.ReflectionPad2d(2), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.utils.spectral_norm(nn.Conv2d(512, 256, 1, bias=True)), nn.LeakyReLU(0.2, True)]
        en3_3 = [nn.ReflectionPad2d(1), nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True), nn.utils.spectral_norm(nn.Conv2d(1024, 256, 1, bias=True)), nn.LeakyReLU(0.2, True)]
        aff1_3 = [nn.Conv2d(256, 1, 1, bias=True)]
        aff2_3 = [nn.Conv2d(256, 1, 1, bias=True)]
        aff3_3 = [nn.Conv2d(256, 1, 1, bias=True)]
		
        self.fc = nn.utils.spectral_norm(nn.Linear(ndf * 2, 1, bias=False))
        self.conv1x1 = nn.Conv2d(ndf * 2, ndf, kernel_size=1, stride=1, bias=True)
        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.lamda = nn.Parameter(torch.zeros(1))
		
        mult = 64
        Dis1_1 = []
        for i in range(3):
          Dis1_1 += [nn.ReflectionPad2d(1),
                          nn.utils.spectral_norm(
                          nn.Conv2d(mult*(2**i), mult*(2**(i+1)), kernel_size=3, stride=2, padding=0, bias=True)),
                          nn.LeakyReLU(0.2, True)]
        Dis1_1 += [nn.utils.spectral_norm(
                          nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0, bias=True))]
                
        
        mult = 128
        Dis2_2 = []
        for i in range(3):
          Dis2_2 += [nn.ReflectionPad2d(1),
                          nn.utils.spectral_norm(
                          nn.Conv2d(mult*(2**i), mult*(2**(i+1)), kernel_size=3, stride=2, padding=0, bias=True)),
                          nn.LeakyReLU(0.2, True)]
        Dis2_2 += [nn.utils.spectral_norm(
                          nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0, bias=True))]
                
        
        mult = 256
        Dis3_3 = []
        for i in range(3):
          Dis3_3 += [nn.ReflectionPad2d(1),
                          nn.utils.spectral_norm(
                          nn.Conv2d(mult*(2**i), mult*(2**(i+1)), kernel_size=3, stride=2, padding=0, bias=True)),
                          nn.LeakyReLU(0.2, True)]
        Dis3_3 += [nn.utils.spectral_norm(
                          nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0, bias=True))]
                



        self.Dis1_1 = nn.Sequential(*Dis1_1)
        self.Dis2_2 = nn.Sequential(*Dis2_2)
        self.Dis3_3 = nn.Sequential(*Dis3_3)
        self.en1_1 = nn.Sequential(*en1_1)
        self.en2_1 = nn.Sequential(*en2_1)
        self.en3_1 = nn.Sequential(*en3_1)
        self.en1_2 = nn.Sequential(*en1_2)
        self.en2_2 = nn.Sequential(*en2_2)
        self.en3_2 = nn.Sequential(*en3_2)
        self.en1_3 = nn.Sequential(*en1_3)
        self.en2_3 = nn.Sequential(*en2_3)
        self.en3_3 = nn.Sequential(*en3_3)
        self.aff1_2 = nn.Sequential(*aff1_2)
        self.aff2_2 = nn.Sequential(*aff2_2)
        self.aff3_2 = nn.Sequential(*aff3_2)
        self.aff1_1 = nn.Sequential(*aff1_1)
        self.aff2_1 = nn.Sequential(*aff2_1)
        self.aff3_1 = nn.Sequential(*aff3_1)
        self.aff1_3 = nn.Sequential(*aff1_3)
        self.aff2_3 = nn.Sequential(*aff2_3)
        self.aff3_3 = nn.Sequential(*aff3_3)

    def forward(self, input):

        layer1out, layer2out, layer3out = pretrain_res(input)
 
        x1_2 = self.en1_2(layer1out)
        x2_2 = self.en2_2(layer2out)
        x3_2 = self.en3_2(layer3out)
        
        x1_2 = x1_2 * self.aff1_2(x1_2)
        x2_2 = x2_2*self.aff2_2(x2_2)
        x3_2 = x3_2*self.aff3_2(x3_2)
        x = x1_2 + x2_2 + x3_2
        
        #fution D1/D3
        x1_1 = self.en1_1(layer1out)
        x2_1 = self.en2_1(layer2out)
        x3_1 = self.en3_1(layer3out)
        
        x1_1 = x1_1*self.aff1_1(x1_1)
        x2_1 = x2_1*self.aff2_1(x2_1)
        x3_1 = x3_1*self.aff3_1(x3_1)
        D1_0 = x1_1 + x2_1 + x3_1
        
        x1_3 = self.en1_3(layer1out)
        x2_3 = self.en2_3(layer2out)
        x3_3 = self.en3_3(layer3out)
        
        x1_3 = x1_3*self.aff1_3(x1_3)
        x2_3 = x2_3*self.aff2_3(x2_3)
        x3_3 = x3_3*self.aff3_3(x3_3)
        D3_0 = x1_3 + x2_3 + x3_3
		
        #x = self.model(input)

        x_0 = x

        gap = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
        x = torch.cat([x, x], 1)
        cam_logit = torch.cat([gap, gmp], 1)
        cam_logit = self.fc(cam_logit.view(cam_logit.shape[0], -1))
        weight = list(self.fc.parameters())[0]
        x = x * weight.unsqueeze(2).unsqueeze(3)
        x = self.conv1x1(x)

        x = self.lamda*x + x_0
        # print("lamda:",self.lamda)

        x = self.leaky_relu(x)
        
        heatmap = torch.sum(x, dim=1, keepdim=True)

        z = x
        out = (torch.mean(self.Dis1_1(D1_0)) + torch.mean(self.Dis2_2(x)) + torch.mean(self.Dis3_3(D3_0)))/3 
        
        return out, cam_logit, heatmap, z

    
class NewResnet(nn.Module):
    def __init__(self, output_layers, *args):
        super().__init__(*args)
        self.output_layers = output_layers
        #print(self.output_layers)
        self.selected_out = OrderedDict()
        #PRETRAINED MODEL
        self.pretrained = models.resnet152(pretrained=True).cuda()
        #self.pretrained = pred
        self.fhooks = []
        

        for i,l in enumerate(list(self.pretrained._modules.keys())):
            if i in self.output_layers:
                self.fhooks.append(getattr(self.pretrained,l).register_forward_hook(self.forward_hook(l)))
                #print(getattr(self.pretrained,l))
        #self.fhooks.append(self.pretrained.layer3[0].downsample[1].register_forward_hook(self.forward_hook(1))

    def forward_hook(self,layer_name):
        def hook(module, input, output):
            self.selected_out[layer_name] = output
        return hook

    def forward(self, x):
        out = self.pretrained(x)
        return self.selected_out

def pretrain_res(x):
        model = NewResnet(output_layers = [0,1,2,3,4,5,6,7,8,9])
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(dev)
        layerout = model(x)
        layer1out = layerout['layer1']
        layer2out = layerout['layer2']
        layer3out = layerout['layer3']

        return layer1out, layer2out, layer3out


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, img_size=256, light=False):
        assert(n_blocks >= 0)
        super(Generator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.img_size = img_size
        self.light = light
		

        mult = 4
        UpBlock0 = [nn.ReflectionPad2d(1),
                    nn.Conv2d(int(ngf * mult / 2), ngf * mult, kernel_size=3, stride=2, padding=0, bias=True),
                    ILN(ngf * mult),
                    nn.ReLU(True)]

        self.relu = nn.ReLU(True)

        # Gamma, Beta block
        #if self.light:
        #    FC = [nn.Linear(ngf * mult, ngf * mult, bias=False),
        #          nn.ReLU(True),
        #          nn.Linear(ngf * mult, ngf * mult, bias=False),
        #          nn.ReLU(True)]
        #else:
        #    FC = [nn.Linear(img_size // mult * img_size // mult * ngf * mult, ngf * mult, bias=False),
        #          nn.ReLU(True),
        #          nn.Linear(ngf * mult, ngf * mult, bias=False),
        #          nn.ReLU(True)]
        #self.gamma = nn.Linear(ngf * mult, ngf * mult, bias=False)
        #self.beta = nn.Linear(ngf * mult, ngf * mult, bias=False)

        # attention Bottleneck
	      #	attention = AttentionBlock(ngf * mult, use_bias=False)
        attention = MultiSelfAttentionBlock(dim = ngf, featur= ngf * mult, n_channel = 8)
        
        UpBlock1 = [nn.ReflectionPad2d(1),
                    nn.Conv2d(ngf * mult, ngf * mult, kernel_size=3, stride=1, padding=0, bias=True),
                    nn.ReLU(True),
                    nn.Conv2d(ngf * mult, ngf * mult, kernel_size=3, stride=1, padding=1, bias=True)]
    
        # Up-Sampling
        UpBlock2_1 = [nn.ReflectionPad2d(1),   
                    nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=0, bias=False),
                    nn.PixelShuffle(2),
                    ILN(int(ngf * mult / 2)),
                    nn.ReLU(True),
                    nn.Conv2d(int(ngf * mult / 2), int(ngf * mult / 4), kernel_size=3, stride=1, bias=True),
                    nn.PixelShuffle(2),
                    ILN(int(ngf * mult / 2)),
                    nn.ReLU(True)
                    ]
        UpBlock2_2 = [nn.ReflectionPad2d(1),   
                    nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=0, bias=False),
                    nn.PixelShuffle(2),
                    ILN(int(ngf * mult / 2)),
                    nn.ReLU(True),
                    nn.Conv2d(int(ngf * mult / 2), int(ngf * mult / 4), kernel_size=3, stride=1, bias=True),
                    nn.PixelShuffle(2),
                    ILN(int(ngf * mult / 2)),
                    nn.ReLU(True)
                    ]

        UpBlock3 = [nn.ReflectionPad2d(3),
                    nn.Conv2d(ngf, output_nc, kernel_size=7, stride=1, padding=0, bias=False),
                    nn.Tanh()]
                    
        fusi = fusions.Block([ngf,ngf], ngf)

        self.UpBlock0 = nn.Sequential(*UpBlock0)
        self.UpBlock1 = nn.Sequential(*UpBlock1)
        self.UpBlock2_1 = nn.Sequential(*UpBlock2_1)
        self.UpBlock2_2 = nn.Sequential(*UpBlock2_2)
        self.UpBlock3 = nn.Sequential(*UpBlock3)

    def forward(self, z):
        x = z
        x = self.UpBlock0(x)
        y = x

        #if self.light:
        #    x_ = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        #    x_ = self.FC(x_.view(x_.shape[0], -1))
        #else:
        #    x_ = self.FC(x.view(x.shape[0], -1))
        #gamma, beta = self.gamma(x_), self.beta(x_)

        for i in range(self.n_blocks):
            x = attention(x)
            #x = attention(x, ngf * mult, gamma, beta, use_bias=False)
            
        x = self.UpBlock1(x)
        x = x + y	
        
        x_de = [self.UpBlock2_1(x), self.UpBlock2_2(x)]
        x = fusi(x_de)

        out = self.UpBlock3(x)

        return out
		
class AttentionBlock(nn.Module):
    def __init__(self, x, dim, use_bias):
        super(AttentionBlock, self).__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm1 = adaILN(dim)
        self.relu1 = nn.ReLU(True)

        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm2 = adaILN(dim)
		
        self.dence2 = nn.Linear(dim,dim/2)
        self.relu2 = nn.ReLU(True)
        self.relu3 = nn.ReLU(True)
        self.dence3 = nn.Linear(dim/2,dim)
        self.conv3 = nn.Conv2d(dim*2, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.avgpool = nn.AvgPool2d(64, stride=1)
		
    def forward(self, x, gamma, beta):
        x1 = self.pad1(x)
        x1 = self.conv1(x1)
        x1 = self.norm1(x1, gamma, beta)
        x1 = self.relu1(x1)
        x1 = self.pad2(x1)
        x1 = self.conv2(x1)
		
        x2 = self.avgpool(x1)
        x2 = self.dence2(x2)
        x2 = self.relu2(x2)
        x2 = self.dence3(x2)
		
        x3 = torch.cat((x1,(x1*x2)),2)
        out = self.conv3(x3)
        out = self.norm2(out, gamma, beta)

        return out + x
		
class MultiSelfAttentionBlock(nn.Module):
    def __init__(self, dim = 256, featur= 64, n_channel = 8):
        super(MultiSelfAttentionBlock, self).__init__()
        self.dim = dim
        self.n_channel = n_channel
        self.featur = featur
        self.atten  = torch.nn.MultiheadAttention(dim, n_channel)
		
		
    def forward(self, x):
        out = torch.reshape(x, (self.featur, self.dim, self.dim))
        out = self.atten(out, out, out)
        out = torch.reshape(out, (1, self.featur, self.dim, self.dim))

        return out + x
		
