import random
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import tifffile as tiff

from micro import TEST,TRAIN

abspath = os.path.abspath(__file__) # 获取所执行脚本的绝对路径    
proj_path = os.path.dirname(abspath) # 获取父级路径            
proj_path = os.path.dirname(proj_path)                          
proj_path = os.path.dirname(proj_path)                           



class ISIC2018_Datasets(Dataset):
    def __init__(self, mode, transformer):
        super().__init__()
        cwd=proj_path+'/datasets'   # 设置数据集的工作目录
        self.mode = mode  # 设置模式（训练或测试）
        
        # 设定真值（Ground Truth）和图像的路径
        gts_path = os.path.join(cwd, 'data', 'ISIC2018', 'ISIC2018_Task1_Training_GroundTruth', 'ISIC2018_Task1_Training_GroundTruth')
        images_path = os.path.join(cwd, 'data', 'ISIC2018', 'ISIC2018_Task1-2_Training_Input', 'ISIC2018_Task1-2_Training_Input')

        # 获取图像文件和真值文件列表
        images_list = sorted(os.listdir(images_path))
        images_list = [item for item in images_list if "jpg" in item]  # 只保留jpg格式的图像
        gts_list = sorted(os.listdir(gts_path))
        gts_list = [item for item in gts_list if "png" in item]  # 只保留png格式的真值图

        self.data = []
        for i in range(len(images_list)):
            image_path = images_path + '/' + images_list[i]  # 构建图像路径
            mask_path = gts_path + '/' + gts_list[i]  # 构建真值路径
            self.data.append([image_path, mask_path])  # 将图像路径和真值路径存入列表
        
        self.transformer = transformer  # 图像变换操作
        random.shuffle(self.data)  # 随机打乱数据顺序
  
        # 根据模式切分数据集
        if mode == TRAIN:
            self.data = self.data[:2075]  # 训练集前2075个样本
        elif mode == TEST:
            self.data = self.data[2075:2594]  # 测试集的样本
        
        self.data_buf = self.cuda_buffer()  # 将数据预加载到CUDA内存中
        print(len(self.data))  # 打印数据集大小

    def getitem_val(self, index):
        # 获取验证集的图像和真值
        image_path, gt_path = self.data[index]  # 根据索引获取路径
        image = Image.open(image_path).convert('RGB')  # 读取图像并转换为RGB模式
        image = np.array(image)  # 转换为NumPy数组
        image = np.transpose(image, axes=(2, 0, 1))  # 调整维度为(C, H, W)
        gt = Image.open(gt_path).convert('L')  # 读取真值图并转换为灰度模式
        gt = np.array(gt)  # 转换为NumPy数组
        gt = np.expand_dims(gt, axis=2) / 255  # 扩展维度并归一化
        gt = np.transpose(gt, axes=(2, 0, 1))  # 调整维度为(C, H, W)
        image, gt = self.transformer((image, gt))  # 应用变换
        
        if self.mode == TEST:
            return image, gt, image_path.split('/')[-1]  # 返回图像、真值和图像名称
        return image, gt  # 返回图像和真值

    # 将数据集加载到内存中，提高训练速度
    def cuda_buffer(self):
        data_buf = []  # 初始化数据缓存
        id = 0  # ID计数器
        for data in self.data:
            image_path, gt_path = data  # 解构路径
            image = Image.open(image_path).convert('RGB')  # 读取图像
            image = np.array(image)  # 转换为NumPy数组
            image = np.transpose(image, axes=(2, 0, 1))  # 调整维度为(C, H, W)
            gt = Image.open(gt_path).convert('L')  # 读取真值图
            gt = np.array(gt)  # 转换为NumPy数组
            gt = np.expand_dims(gt, axis=2) / 255  # 扩展维度并归一化
            gt = np.transpose(gt, axes=(2, 0, 1))  # 调整维度为(C, H, W)
            image, gt = self.transformer((image, gt))  # 应用变换
            
            # 将数据转移到CUDA设备上
            image = image.cpu()
            gt = gt.cpu()    
            
            if self.mode == TEST:
                data_buf.append([image, gt, image_path.split('/')[-1]])  # 缓存测试数据
            else:
                data_buf.append([image, gt])  # 缓存训练数据
            
            # 每处理20个样本打印一次进度
            if id % 20 == 0:
                print(id)
            id += 1
        
        return data_buf  # 返回数据缓存

    def __getitem__(self, index):
        # 根据索引返回样本
        if self.mode != TEST:
            image, gt = self.data_buf[index]  # 从缓存中获取训练数据
            image = image.cpu()  # 转回CPU
            gt = gt.cpu()  # 转回CPU
            return image, gt  # 返回图像和真值
        else:
            image, gt, image_name = self.data_buf[index]  # 从缓存中获取测试数据
            image = image.cpu()  # 转回CPU
            gt = gt.cpu()  # 转回CPU
            return image, gt, image_name  # 返回图像、真值和图像名称

    def __len__(self):
        # 返回数据集大小
        return len(self.data)


class PH2_Datasets(Dataset):
    def __init__(self,mode,transformer):
        super().__init__()
        self.mode=mode
        cwd=proj_path+'/datasets'
        images_path=os.path.join(cwd,'Medical_image','PH2','PH2Dataset','PH2 Dataset images')
        images_list=sorted(os.listdir(images_path))
        random.shuffle(images_list)
        self.data=[]
        for path in images_list:
            image_path=os.path.join(images_path,path,path+'_Dermoscopic_Image',path+'.bmp')
            gt_path=os.path.join(images_path,path,path+'_lesion',path+'_lesion.bmp')
            self.data.append([image_path, gt_path])
        self.data=self.data[0:100]
        self.transformer=transformer
        print(f'the length of datasets is {len(self.data)}')

    def __getitem__(self, index):
        image_path, gt_path=self.data[index]
        image = Image.open(image_path).convert('RGB')
        image=np.array(image)
        image = np.transpose(image, axes=(2, 0, 1))
        gt = Image.open(gt_path).convert('L')
        gt = np.array(gt)
        gt=np.expand_dims(gt, axis=2) / 255
        gt = np.transpose(gt, axes=(2, 0, 1))
        image, gt = self.transformer((image, gt))
        if self.mode==TEST:
            return image,gt,image_path.split('/')[-1]
        return image,gt
    
    def __len__(self):
        return len(self.data)




class BUSI_alter_Datasets(Dataset):
    def __init__(self,mode,transformer):
        super().__init__()
        self.mode=mode
        cwd=proj_path+'/datasets'
        data_path_1=os.path.join(cwd,'Medical_image','BUSI_alter','Dataset_BUSI','Dataset_BUSI_with_GT','benign')
        data_path_2=os.path.join(cwd,'Medical_image','BUSI_alter','Dataset_BUSI','Dataset_BUSI_with_GT','malignant')
        data_path_3=os.path.join(cwd,'Medical_image','BUSI_alter','Dataset_BUSI','Dataset_BUSI_with_GT','normal')

        benign_list=sorted(os.listdir(data_path_1))
        malignant_list=sorted(os.listdir(data_path_2))
        norm_list=sorted(os.listdir(data_path_3))

        benign_image_list=[item for item in benign_list if ").png" in item]
        benign_gt_list=[item for item in benign_list if "mask.png" in item]

        malignant_image_list=[item for item in malignant_list if ").png" in item]
        malignant_gt_list=[item for item in malignant_list if "mask.png" in item]

        norm_image_list=[item for item in norm_list if ").png" in item]
        norm_gt_list=[item for item in norm_list if "mask.png" in item]
        print(len(norm_gt_list))
        
        self.data=[]
        for i in range(len(benign_image_list)):
            image_path=data_path_1+'/'+benign_image_list[i]
            mask_path=data_path_1+'/'+benign_gt_list[i]
            self.data.append([image_path, mask_path])
        for i in range(len(malignant_image_list)):
            image_path=data_path_2+'/'+malignant_image_list[i]
            mask_path=data_path_2+'/'+malignant_gt_list[i]
            self.data.append([image_path, mask_path])
        for i in range(len(norm_image_list)):
            image_path=data_path_3+'/'+norm_image_list[i]
            mask_path=data_path_3+'/'+norm_gt_list[i]
            self.data.append([image_path, mask_path])


        if mode==TRAIN:
            self.data=self.data[:685]
        if mode==TEST:
            self.data=self.data[685:857]
        
        self.transformer=transformer
        print(len(self.data))


    def __getitem__(self, index):
        image_path, gt_path=self.data[index]
        image = Image.open(image_path).convert('RGB')
        image=np.array(image)
        image = np.transpose(image, axes=(2, 0, 1))
        gt = Image.open(gt_path).convert('L')
        gt = np.array(gt)
        gt=np.expand_dims(gt, axis=2) / 255
        gt = np.transpose(gt, axes=(2, 0, 1))
        image, gt = self.transformer((image, gt))
        if self.mode==TEST:
            return image,gt,image_path.split('/')[-1]
        return image,gt
    
    def __len__(self):
        return len(self.data)

class BUSI_Datasets(Dataset):
    def __init__(self,mode,transformer):
        super().__init__()
        self.mode=mode
        cwd=proj_path+'/datasets'
        data_path_1=os.path.join(cwd,'Medical_image','BUSI','Dataset_BUSI','Dataset_BUSI_with_GT','benign')
        data_path_2=os.path.join(cwd,'Medical_image','BUSI','Dataset_BUSI','Dataset_BUSI_with_GT','malignant')
        data_path_3=os.path.join(cwd,'Medical_image','BUSI','Dataset_BUSI','Dataset_BUSI_with_GT','normal')

        benign_list=sorted(os.listdir(data_path_1))
        malignant_list=sorted(os.listdir(data_path_2))
        norm_list=sorted(os.listdir(data_path_3))

        benign_image_list=[item for item in benign_list if ").png" in item]
        benign_gt_list=[item for item in benign_list if "mask.png" in item]

        malignant_image_list=[item for item in malignant_list if ").png" in item]
        malignant_gt_list=[item for item in malignant_list if "mask.png" in item]

        norm_image_list=[item for item in norm_list if ").png" in item]
        norm_gt_list=[item for item in norm_list if "mask.png" in item]
        print(len(norm_gt_list))
        
        self.data=[]
        for i in range(len(benign_image_list)):
            image_path=data_path_1+'/'+benign_image_list[i]
            mask_path=data_path_1+'/'+benign_gt_list[i]
            self.data.append([image_path, mask_path])
        for i in range(len(malignant_image_list)):
            image_path=data_path_2+'/'+malignant_image_list[i]
            mask_path=data_path_2+'/'+malignant_gt_list[i]
            self.data.append([image_path, mask_path])
        for i in range(len(norm_image_list)):
            image_path=data_path_3+'/'+norm_image_list[i]
            mask_path=data_path_3+'/'+norm_gt_list[i]
            self.data.append([image_path, mask_path])


        if mode==TRAIN:
            self.data=self.data[:685]
        elif mode==TEST:
            self.data=self.data[685:857]
        
        self.transformer=transformer
        print(len(self.data))


    def __getitem__(self, index):
        image_path, gt_path=self.data[index]
        image = Image.open(image_path).convert('RGB')
        image=np.array(image)
        image = np.transpose(image, axes=(2, 0, 1))
        gt = Image.open(gt_path).convert('L')
        gt = np.array(gt)
        gt=np.expand_dims(gt, axis=2) / 255
        gt = np.transpose(gt, axes=(2, 0, 1))
        image, gt = self.transformer((image, gt))
        if self.mode==TEST:
            return image,gt,image_path.split('/')[-1]
        return image,gt
    
    def __len__(self):
        return len(self.data)


class Kvasir_Datasets(Dataset):
    def __init__(self,mode,transformer):
        super().__init__()
        cwd=proj_path+'/datasets'
        self.mode=mode
        gts_path=os.path.join(cwd,'Medical_image','Kvasir','kvasir-seg','Kvasir-SEG','masks')
        images_path=os.path.join(cwd,'Medical_image','Kvasir','kvasir-seg','Kvasir-SEG','images')

        images_list=sorted(os.listdir(images_path))
        images_list = [item for item in images_list if "jpg" in item]
        gts_list=sorted(os.listdir(gts_path))
        gts_list = [item for item in gts_list if "jpg" in item]
        self.data=[]
        for i in range(len(images_list)):
            image_path=images_path+'/'+images_list[i]
            mask_path=gts_path+'/'+gts_list[i]
            self.data.append([image_path, mask_path])
        self.transformer=transformer
        random.shuffle(self.data)
        if mode==TRAIN:
            self.data=self.data[:880]
        elif mode==TEST:
            self.data=self.data[880:1000]

        print(len(self.data))


    def __getitem__(self, index):
        image_path, gt_path=self.data[index]
        image = Image.open(image_path).convert('RGB')
        image=np.array(image)
        image = np.transpose(image, axes=(2, 0, 1))
        gt = Image.open(gt_path).convert('L')
        gt = np.array(gt)
        gt=np.expand_dims(gt, axis=2) / 255
        gt = np.transpose(gt, axes=(2, 0, 1))
        image, gt = self.transformer((image, gt))
        if self.mode==TEST:
            return image,gt,image_path.split('/')[-1]
        return image,gt
    

    def __len__(self):
        return len(self.data)


# class COVID_19_Datasets(Dataset):
#     def __init__(self, mode, transformer):
#         super().__init__()
#         cwd=proj_path+'/datasets'  # 设置数据集的工作目录
#         self.mode = mode  # 设置模式（训练或测试）
        
#         # 设定真值（Ground Truth）和图像的路径
#         gts_path=os.path.join(cwd,'Medical_image','COVID_19','COVID-19_Lung_Infection_train','COVID-19_Lung_Infection_train','masks')
#         images_path=os.path.join(cwd,'Medical_image','COVID_19','COVID-19_Lung_Infection_train','COVID-19_Lung_Infection_train','images')

#         # 获取图像文件和真值文件列表
#         images_list = sorted(os.listdir(images_path))
#         images_list = [item for item in images_list if "jpg" in item]  # 只保留jpg格式的图像
#         gts_list = sorted(os.listdir(gts_path))
#         gts_list = [item for item in gts_list if "png" in item]  # 只保留png格式的真值图

#         self.data = []
#         for i in range(len(images_list)):
#             image_path = images_path + '/' + images_list[i]  # 构建图像路径
#             mask_path = gts_path + '/' + gts_list[i]  # 构建真值路径
#             self.data.append([image_path, mask_path])  # 将图像路径和真值路径存入列表
        
#         self.transformer = transformer  # 图像变换操作
#         random.shuffle(self.data)  # 随机打乱数据顺序
  
#         # 根据模式切分数据集
#         if mode == TRAIN:
#             self.data = self.data[:2075]  # 训练集前2075个样本
#         elif mode == TEST:
#             self.data = self.data[2075:2594]  # 测试集的样本
        
#         self.data_buf = self.cuda_buffer()  # 将数据预加载到CUDA内存中
#         print(len(self.data))  # 打印数据集大小

#     def getitem_val(self, index):
#         # 获取验证集的图像和真值
#         image_path, gt_path = self.data[index]  # 根据索引获取路径
#         image = Image.open(image_path).convert('RGB')  # 读取图像并转换为RGB模式
#         image = np.array(image)  # 转换为NumPy数组
#         image = np.transpose(image, axes=(2, 0, 1))  # 调整维度为(C, H, W)
#         gt = Image.open(gt_path).convert('L')  # 读取真值图并转换为灰度模式
#         gt = np.array(gt)  # 转换为NumPy数组
#         gt = np.expand_dims(gt, axis=2) / 255  # 扩展维度并归一化
#         gt = np.transpose(gt, axes=(2, 0, 1))  # 调整维度为(C, H, W)
#         image, gt = self.transformer((image, gt))  # 应用变换
        
#         if self.mode == TEST:
#             return image, gt, image_path.split('/')[-1]  # 返回图像、真值和图像名称
#         return image, gt  # 返回图像和真值

#     # 将数据集加载到内存中，提高训练速度
#     def cuda_buffer(self):
#         data_buf = []  # 初始化数据缓存
#         id = 0  # ID计数器
#         for data in self.data:
#             image_path, gt_path = data  # 解构路径
#             image = Image.open(image_path).convert('RGB')  # 读取图像
#             image = np.array(image)  # 转换为NumPy数组
#             image = np.transpose(image, axes=(2, 0, 1))  # 调整维度为(C, H, W)
#             gt = Image.open(gt_path).convert('L')  # 读取真值图
#             gt = np.array(gt)  # 转换为NumPy数组
#             gt = np.expand_dims(gt, axis=2) / 255  # 扩展维度并归一化
#             gt = np.transpose(gt, axes=(2, 0, 1))  # 调整维度为(C, H, W)
#             image, gt = self.transformer((image, gt))  # 应用变换
            
#             # 将数据转移到CUDA设备上
#             image = image.to('cuda:0')
#             gt = gt.to('cuda:0')
            
#             if self.mode == TEST:
#                 data_buf.append([image, gt, image_path.split('/')[-1]])  # 缓存测试数据
#             else:
#                 data_buf.append([image, gt])  # 缓存训练数据
            
#             # 每处理20个样本打印一次进度
#             if id % 20 == 0:
#                 print(id)
#             id += 1
        
#         return data_buf  # 返回数据缓存

#     def __getitem__(self, index):
#         # 根据索引返回样本
#         if self.mode != TEST:
#             image, gt = self.data_buf[index]  # 从缓存中获取训练数据
#             image = image.cpu()  # 转回CPU
#             gt = gt.cpu()  # 转回CPU
#             return image, gt  # 返回图像和真值
#         else:
#             image, gt, image_name = self.data_buf[index]  # 从缓存中获取测试数据
#             image = image.cpu()  # 转回CPU
#             gt = gt.cpu()  # 转回CPU
#             return image, gt, image_name  # 返回图像、真值和图像名称

#     def __len__(self):
#         # 返回数据集大小
#         return len(self.data)

class COVID_19_Datasets(Dataset):
    def __init__(self,mode,transformer):
        super().__init__()
        cwd=proj_path+'/datasets'
        self.mode=mode
        gts_path=os.path.join(cwd,'Medical_image','COVID_19','COVID-19_Lung_Infection_train','COVID-19_Lung_Infection_train','masks')
        images_path=os.path.join(cwd,'Medical_image','COVID_19','COVID-19_Lung_Infection_train','COVID-19_Lung_Infection_train','images')
        images_list=sorted(os.listdir(images_path))
        images_list = [item for item in images_list if "jpg" in item]
        gts_list=sorted(os.listdir(gts_path))
        gts_list = [item for item in gts_list if "png" in item]

        self.data=[]
        for i in range(len(images_list)):
            image_path=os.path.join(images_path,images_list[i])
            mask_path=os.path.join(gts_path,gts_list[i])
            self.data.append([image_path, mask_path])
        self.transformer=transformer
        random.shuffle(self.data)

        if mode==TRAIN:
            self.data=self.data[:716]
        elif mode==TEST:
            self.data=self.data[716:894]
        print(len(self.data))

    def __getitem__(self, index):
        image_path, gt_path=self.data[index]
        image = Image.open(image_path).convert('RGB')
        image=np.array(image)
        image = np.transpose(image, axes=(2, 0, 1))
        gt = Image.open(gt_path).convert('L')
        gt = np.array(gt)
        gt=np.expand_dims(gt, axis=2) / 255
        gt = np.transpose(gt, axes=(2, 0, 1))
        image, gt = self.transformer((image, gt))
        if self.mode==TEST:
            return image,gt,image_path.split('/')[-1]
        return image,gt
    

    def __len__(self):
        return len(self.data)

class CVC_ClinkDB_Datasets(Dataset):
    def __init__(self,mode,transformer):
        super().__init__()
        cwd=proj_path+'/datasets'
        self.mode=mode
        gts_path=os.path.join(cwd,'Medical_image','CVC_ClinkDB','archive','PNG','Ground Truth')
        images_path=os.path.join(cwd,'Medical_image','CVC_ClinkDB','archive','PNG','Original')
        
        images_list=sorted(os.listdir(images_path))
        images_list = [item for item in images_list if "png" in item]
        gts_list=sorted(os.listdir(gts_path))
        gts_list = [item for item in gts_list if "png" in item]

        self.data=[]
        for i in range(len(images_list)):
            image_path=os.path.join(images_path,images_list[i])
            mask_path=os.path.join(gts_path,gts_list[i])
            self.data.append([image_path, mask_path])
        self.transformer=transformer
        random.shuffle(self.data)
        print(len(self.data))
       
   

    def __getitem__(self, index):
        image_path, gt_path=self.data[index]
        image = Image.open(image_path).convert('RGB')
        image=np.array(image)
        image = np.transpose(image, axes=(2, 0, 1))
        gt = Image.open(gt_path).convert('L')
        gt = np.array(gt)
        gt=np.expand_dims(gt, axis=2) / 255
        gt = np.transpose(gt, axes=(2, 0, 1))
        image, gt = self.transformer((image, gt))
        if self.mode==TEST:
            return image,gt,image_path.split('/')[-1]
        return image,gt
    

    def __len__(self):
        return len(self.data)


class Monu_Seg_Datasets(Dataset):
    def __init__(self,mode,transformer):
        super().__init__()
        cwd=proj_path+'/datasets'
        self.mode=mode

        gts_path=os.path.join(cwd,'Medical_image','Monu_Seg','archive','kmms_test','kmms_test','masks')
        images_path=os.path.join(cwd,'Medical_image','Monu_Seg','archive','kmms_test','kmms_test','images')
     
        images_list=sorted(os.listdir(images_path))
        images_list = [item for item in images_list if "png" in item]
        gts_list=sorted(os.listdir(gts_path))
        gts_list = [item for item in gts_list if "png" in item]

        self.data=[]
        for i in range(len(images_list)):
            image_path=os.path.join(images_path,images_list[i])
            mask_path=os.path.join(gts_path,gts_list[i])
            self.data.append([image_path, mask_path])
        
        gts_path_=os.path.join(cwd,'Medical_image','Monu_Seg','archive','kmms_training','kmms_training','masks')
        images_path_=os.path.join(cwd,'Medical_image','Monu_Seg','archive','kmms_training','kmms_training','images')

        images_list_=sorted(os.listdir(images_path_))
        images_list_ = [item for item in images_list_ if "tif" in item]
        gts_list_=sorted(os.listdir(gts_path_))
        gts_list_ = [item for item in gts_list_ if "png" in item]

        
        for i in range(len(images_list_)):
            image_path=os.path.join(images_path_,images_list_[i])
            mask_path=os.path.join(gts_path_,gts_list_[i])
            self.data.append([image_path, mask_path])
        
        self.transformer=transformer
        random.shuffle(self.data)
        if mode==TRAIN:
            self.data=self.data[:59]
        elif mode==TEST:
            self.data=self.data[59:74]
        print(len(self.data))
        
        

    def __getitem__(self, index):
        image_path, gt_path=self.data[index]
        if '.tif' in image_path:
            image = tiff.imread(image_path)
        else:
            image = Image.open(image_path).convert('RGB')
        image=np.array(image)
        image = np.transpose(image, axes=(2, 0, 1))
        gt = Image.open(gt_path).convert('L')
        gt = np.array(gt)
        gt=np.expand_dims(gt, axis=2) / 255
        gt = np.transpose(gt, axes=(2, 0, 1))
        image, gt = self.transformer((image, gt))
        if self.mode==TEST:
            return image,gt,image_path.split('/')[-1]
        return image,gt
    

    def __len__(self):
        return len(self.data)
