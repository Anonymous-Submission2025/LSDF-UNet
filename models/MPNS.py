import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import models.hotfeat as hotfeat

class FirstOctaveConv(nn.Module):   
      
    def __init__(self, in_channels, out_channels,kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(FirstOctaveConv, self).__init__()
        self.stride = stride
        kernel_size = kernel_size[0]   # 3
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.h2l = torch.nn.Conv2d(in_channels, int(alpha * in_channels), # (512,256)
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = torch.nn.Conv2d(in_channels, in_channels - int(alpha * in_channels), 
                                   kernel_size, 1, padding, dilation, groups, bias)

    def forward(self, x):    # x：n,c,h,w
        if self.stride ==2:
            x = self.h2g_pool(x)

        X_h2l = self.h2g_pool(x) 
        X_h = x
        X_h = self.h2h(X_h)  
        X_l = self.h2l(X_h2l) 

        return X_h, X_l

class OctaveConv(nn.Module): # 低、高频输入，低、高频输出 对应第二个红色和绿色
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(OctaveConv, self).__init__()
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.stride = stride
        
        self.l2l = torch.nn.Conv2d(int(alpha * in_channels), int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        
        
        self.l2h = torch.nn.Conv2d(int(alpha * in_channels), out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        
       
        self.h2l = torch.nn.Conv2d(in_channels - int(alpha * in_channels), int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        
        
        self.h2h = torch.nn.Conv2d(in_channels - int(alpha * in_channels),
                                   out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)

    def forward(self, x):
        X_h, X_l = x

        if self.stride == 2:
            X_h, X_l = self.h2g_pool(X_h), self.h2g_pool(X_l)

        X_h2l = self.h2g_pool(X_h)

        X_h2h = self.h2h(X_h)
        X_l2h = self.l2h(X_l)

        X_l2l = self.l2l(X_l)
        X_h2l = self.h2l(X_h2l)

        X_l2h = F.interpolate(X_l2h, (int(X_h2h.size()[2]),int(X_h2h.size()[3])), mode='bilinear')

        X_h = X_l2h + X_h2h
        X_l = X_h2l + X_l2l

        return X_h, X_l

class LastOctaveConv(nn.Module): # 低频和高频对齐输出
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(LastOctaveConv, self).__init__()   # 继承 nn.Module 的一些属性和方法
        self.stride = stride
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)

        self.l2h = torch.nn.Conv2d(int(alpha * out_channels), out_channels,
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = torch.nn.Conv2d(out_channels - int(alpha * out_channels),
                                   out_channels,
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
    def forward(self, x):
        X_h, X_l = x

        if self.stride == 2:
            X_h, X_l = self.h2g_pool(X_h), self.h2g_pool(X_l)

        X_h2h = self.h2h(X_h)
        X_l2h = self.l2h(X_l) 
        
        X_l2h = F.interpolate(X_l2h, (int(X_h2h.size()[2]), int(X_h2h.size()[3])), mode='bilinear') 
        # print("X_l2h.shape:", X_l2h.shape)
        # print("X_h2h.shape:", X_h2h.shape)
        # X_h = X_h2h + X_l2h  
        return X_h2h,X_l2h       

# Frequency-aware module(FAM)
class Octave(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3)):
        super(Octave, self).__init__()
       
        self.fir = FirstOctaveConv(in_channels, in_channels, kernel_size)
        self.mid1 = OctaveConv(in_channels, in_channels, kernel_size)   
        self.mid2 = OctaveConv(in_channels, out_channels, kernel_size)  

        
        self.lst = LastOctaveConv(in_channels, out_channels, kernel_size)

    def forward(self, x):   
        x0 = x
        # print("x0.shape:", x0.shape)

        x_h, x_l = self.fir(x)    
        # print("x_h.shape:", x_h.shape)
        # print("x_l.shape:", x_l.shape)               
        x_hh, x_ll = x_h, x_l,
        # x_1 = x_hh +x_ll
        x_h_1, x_l_1 = self.mid1((x_h, x_l))     
        x_h_2, x_l_2 = self.mid1((x_h_1, x_l_1)) 
        x_h_5, x_l_5 = self.mid2((x_h_2, x_l_2)) 

        x_FH, x_FL = self.lst((x_h_5, x_l_5)) 
        # print("x_FH.shape:", x_FH.shape)
        # print("x_FL.shape:", x_FL.shape)
        return x_FH, x_FL



class DPFA(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias,patch_sizes):    # dim = 32, ffn_expansion_factor = 4, bias = True
        super(DPFA, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)           # 4 * 32 = 128

        self.ffn_expansion_factor = ffn_expansion_factor
        self.bias = bias
        self.patch_size1 = patch_sizes[0]
        self.patch_size2 = patch_sizes[1]
        self.bias = bias
        self.dim = dim

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.fft1 = nn.Parameter(torch.ones((dim, 1, 1, self.patch_size1, self.patch_size1 // 2 + 1)))
        self.fft2 = nn.Parameter(torch.ones((dim, 1, 1, self.patch_size2, self.patch_size2 // 2 + 1)))
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        
        self.fab = Octave(dim, dim, kernel_size=(3, 3))


    def fft_process(self, x, patch_size1, patch_size2):
        x1_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size1,
                            patch2=self.patch_size1)
        # x_patch : (B, dim, H, W, patch_size, patch_size)
        x1_patch_fft = torch.fft.rfft2(x1_patch.float())  
        # x_patch_fft : (B, dim, H, W, patch_size, patch_size // 2 + 1)
        x1_patch_fft = x1_patch_fft * self.fft1
        # x_patch_fft : (B, dim, H, W, patch_size, patch_size // 2 + 1)
        x1_patch = torch.fft.irfft2(x1_patch_fft, s=(self.patch_size1, self.patch_size1))
        # x_patch : (B, dim, H, W, patch_size, patch_size)
        x1 = rearrange(x1_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size1,
                      patch2=self.patch_size1)
        return x1

    def forward(self, x):
        
        x = self.project_in(x)  
        
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        
        x = F.gelu(x1) * x2         
        
        x = self.project_out(x)

        # print("x",x.shape)
        x_h, x_l = self.fab(x)


        x1=self.fft_process(x_h,self.patch_size1,self.patch_size2)

        x2=self.fft_process(x_l,self.patch_size2,self.patch_size2)
        x = x1 + x2
       
        return x




