import torch.nn as nn

from .backbone import enconder1,enconder2_3,enconder4
from .backbone import decoder_1_2_3,decoder4
from .MPNS import DPFA


class LSDF_Unet(nn.Module):
    def __init__(self,input_channels=3, out_channels:list=None,kernel_list=None):
        super().__init__()
        #encoding
        # self.en1=DFFM(out_channels[0],out_channels[1],sample=True,up=False,kernel_list=kernel_list,patchsizes=[32,128])
        # self.en2=DFFM(out_channels[1],out_channels[2],sample=True,up=False,kernel_list=kernel_list,patchsizes=[16,64])
        # self.en3=DFFM(out_channels[2],out_channels[3],sample=True,up=False,kernel_list=kernel_list,patchsizes=[8,32])
        # self.en4=DFFM(out_channels[3],out_channels[4],sample=True,up=False,kernel_list=kernel_list,patchsizes=[4,16])

        self.en1=enconder1(out_channels[0],out_channels[1])
        self.en2=enconder2_3(out_channels[1],out_channels[2])
        self.en3=enconder2_3(out_channels[2],out_channels[3])
        self.en4=enconder4(out_channels[3],out_channels[4])


        #decoding
        self.de1=decoder_1_2_3(out_channels[1],out_channels[0])
        self.de2=decoder_1_2_3(out_channels[2],out_channels[1])
        self.de3=decoder_1_2_3(out_channels[3],out_channels[2])
        self.de4=decoder4(out_channels[4],out_channels[3])

        #patch
        self.patch_conv=nn.Sequential(
            nn.Conv2d(input_channels,out_channels[0],3,padding=1),
            nn.BatchNorm2d(out_channels[0])
        )

        #swin
        self.swim1 = DPFA(out_channels[1],ffn_expansion_factor=4,bias=True,patch_sizes=[16,64])
        self.swim2 = DPFA(out_channels[2],ffn_expansion_factor=4,bias=True,patch_sizes=[8,32])
        self.swim3 = DPFA(out_channels[3],ffn_expansion_factor=4,bias=True,patch_sizes=[4,16])
        


        #prediction
        self.ph=PH(out_channels)
        
    def forward(self,x):
        #patch
        x=self.patch_conv(x)

        #encoding
        e1=self.en1(x)
        e2=self.en2(e1)
        e3=self.en3(e2)
        e4=self.en4(e3)


        #swin
        e1=self.swim1(e1)
        e2=self.swim2(e2)
        e3=self.swim3(e3)


        #decoding
        d4=self.de4(e4)
        d3=self.de3(d4+e3)
        d2=self.de2(d3+e2)
        d1=self.de1(d2+e1)
        

        x_pre=self.ph([d4,d3,d2,d1])
        return x_pre



class PH_Block(nn.Module):
    def __init__(self,in_channels,scale_factor=1):
        super().__init__()
        if scale_factor>1:
            self.upsample=nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        else:
            self.upsample=None
        self.pro=nn.Conv2d(in_channels,1,1)
        self.sig=nn.Sigmoid()

    def forward(self,x):
        if self.upsample!=None:
            x=self.upsample(x)
        x=self.pro(x)
        x=self.sig(x)
        return x

class PH(nn.Module):
    def __init__(self,in_channels=[12,24,36,48],scale_factor=[1,2,4,8]):
        super().__init__()
        self.ph1=PH_Block(in_channels[0],scale_factor[0])
        self.ph2=PH_Block(in_channels[1],scale_factor[1])
        self.ph3=PH_Block(in_channels[2],scale_factor[2])
        self.ph4=PH_Block(in_channels[3],scale_factor[3])
        
    def forward(self,x):
        x4,x3,x2,x1=x
        x1=self.ph1(x1)
        x2=self.ph2(x2)
        x3=self.ph3(x3)
        x4=self.ph4(x4)
        return [x1,x2,x3,x4]
