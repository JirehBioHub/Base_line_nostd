import os
import sys
import numpy as np
from PIL import Image
import cv2
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import  SummaryWriter
from torchvision import transforms, datasets

from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from collections import OrderedDict
from tqdm import  tqdm

image_train_work_dir = '/home/ncp/workspace/202002n049/049'
label_train_work_dir = '/home/ncp/workspace/test/1.Training/filtering_train_label/'

image_vaild_work_dir = '/home/ncp/workspace/202002n049/049'
label_valid_work_dir = '/home/ncp/workspace/test/1.Training/filtering_valid_label/'

train_name = 'FPN-effb5-StepLR-0001-0997-FT-Adam-4-23-train_155ret1-MB2SSR40GIE7II3_.50.26'
log_dir = '/home/ncp/workspace/test/5. BaseLine/logs/' + train_name
ckpt_dir = '/home/ncp/workspace/test/5. BaseLine/logs/' + train_name+'/ckp'
ckpt_dir = '/home/ncp/workspace/test/5. BaseLine/logs/' + train_name+'/no_std'
ckpt_dir = '/home/ncp/workspace/test/5. BaseLine/Caries/Model_output/'

retrain = False
retrain_ckpt_dir = '/home/ncp/workspace/test/5. BaseLine/Caries/Model_output'

lr = 0.001
batch_size = 1
num_epoch = 2000

os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory.' + directory)

class UNet(nn.Module):
    @property
    def __init__(self):
        super(UNet, self).__init__()

        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                 stride=stride, padding=padding, bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]
            cbr = nn.Sequential(*layers)

            return cbr

        self.enc1_1 = CBR2d(in_channels=1, out_channels=64)
        self.enc1_2 = CBR2d(in_channels=64, out_channels=64)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = CBR2d(in_channels=64, out_channels=128)
        self.enc2_2 = CBR2d(in_channels=128, out_channels=128)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = CBR2d(in_channels=128, out_channels=256)
        self.enc3_2 = CBR2d(in_channels=256, out_channels=256)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4_1 = CBR2d(in_channels=256, out_channels=512)
        self.enc4_2 = CBR2d(in_channels=523, out_channels=512)

        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc5_1 = CBR2d(in_channels=512, out_channels=1024)

        self.dec5_1 = CBR2d(in_channels=1024, out_channels=512)

        self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=2,
                                          stride=2, padding=0, bias=0)

        self.dec4_2 = CBR2d(in_channels=512 * 2, out_channels=512)
        self.dec4_1 = CBR2d(in_channels=512, out_channels=256)

        self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2,
                                          stride=2, padding=0, bias=True)
        self.dec3_2 = CBR2d(in_channels=256 * 2, out_channels=256)
        self.dec3_1 = CBR2d(in_channels=256, out_channels=128)

        self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2,
                                          stride=2, padding=0, bias=True)
        self.dec2_2 = CBR2d(in_channels=128 * 2, out_channels=128)
        self.dec2_1 = CBR2d(in_channels=128, out_channels=64)

        self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2,
                                          stride=2, padding=0, bias=True)

        self.dec1_2 = CBR2d(in_channels=64 * 2, out_channels=64)
        self.dec1_1 = CBR2d(in_channels=64, out_channels=64)

        self.fc = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)
        dec5_1 = self.dec5_1(enc5_1)

        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, enc4_2), dim=1)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)

        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        x = self.fc(dec1_1)
        return x

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class Recurrent_block(nn.Module):
    def __init__(self,ch_out,t=2):
        super(Recurrent_block,self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        for i in range(self.t):

            if i==0:
                x1 = self.conv(x)

            x1 = self.conv(x+x1)
        return  x1

class RRCNN_block(nn.Module):
    def __init__(self,ch_in,ch_out,t=2):
        super(RRCNN_block,self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out,t=t),
            Recurrent_block(ch_out,t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self,x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return  x+x1

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
        nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi

class Attu_Net(nn.Module):
    def __init__(self,img_ch=1,output_ch=1):
        super(Attu_Net,self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)


    def forward(self,x):
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        d5 = self.Up5(x)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4,d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3,d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2,d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1,d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

class R2AttU_Net(nn.Module):
    def __init__(self,img_ch=1,output_ch=1,t=2):
        super(R2AttU_Net,self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch, ch_out=64, t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64, ch_out=128, t=t)

        self.RRCNN3 = RRCNN_block(ch_in=128, ch_out=256, t=t)

        self.RRCNN4 = RRCNN_block(ch_in=256, ch_out=512, t=t)

        self.RRCNN5 = RRCNN_block(ch_in=512, ch_out=1024, t=t)


        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512, t=t)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256, t=t)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128, t=t)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64, t=t)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)


    def forward(self,x):
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4,d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3,d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        x3 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2,d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1,d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

def crop_expand(img, hei_threshold, wid_threshold):
    img_hei, img_wid = img.shape
    cp_img = img.copy()


    crop_hei = (img_hei-hei_threshold)  // 2
    if img_hei-crop_hei-crop_hei > hei_threshold:
        hei_add = img_hei-crop_hei-crop_hei - hei_threshold
        cp_img = cp_img[crop_hei+hei_add:img_hei-crop_hei, :]

    elif img_hei-crop_hei-crop_hei < hei_threshold:
        hei_minus = hei_threshold - img_hei-crop_hei-crop_hei
        cp_img = cp_img[crop_hei-hei_minus:img_hei-crop_hei, :]

    elif img_hei-crop_hei-crop_hei == hei_threshold:
        cp_img = cp_img[crop_hei:img_hei-crop_hei, :]


    crop_wid = (img_wid-wid_threshold) // 2
    if img_wid-crop_wid-crop_wid > wid_threshold:
        hei_add = img_wid-crop_wid-crop_wid - wid_threshold
        cp_img = cp_img[:, crop_wid+hei_add:img_wid-crop_wid]

    elif img_wid-crop_wid-crop_wid < wid_threshold:
        hei_minus = wid_threshold - img_wid-crop_wid-crop_wid
        cp_img = cp_img[:, crop_wid-hei_minus:img_wid-crop_wid]

    elif img_wid-crop_wid-crop_wid == wid_threshold:
        cp_img = cp_img[:, crop_wid:img_wid-crop_wid]

    return cp_img

def fit_8_(img, pooling_cnt=4, fit_num=8):
    img_hei, img_wid = img.shape

    fitted_num_hei = int(str(img_hei)[-2:])
    if fitted_num_hei - fit_num < 0:
        dst_img_hei = img_hei-(10+(fitted_num_hei-fit_num))
    elif fitted_num_hei - fit_num > 0:
        dst_img_hei = img_hei - (fitted_num_hei-fit_num)
    elif fitted_num_hei - fit_num == 0:
        dst_img_hei = img_hei


    fitted_num_wid = int(str(img_wid)[-1])
    if fitted_num_wid - fit_num < 0:
        dst_img_wid = img_wid-(10+(fitted_num_wid-fit_num))
    elif fitted_num_wid - fit_num > 0:
        dst_img_wid = img_wid-(fitted_num_wid-fit_num)
    elif fitted_num_wid - fit_num == 0:
        dst_img_wid = img_wid
    print('dst_img_hei, dst_img_wid = ', dst_img_hei, dst_img_wid)
    print('img_hei, img_wid =', img_hei, img_wid)
    fited_img = img[img_hei-dst_img_hei:, img_wid-dst_img_wid:]
    print("after = ", fited_img.shape)
    return fited_img
def fit_8(img, pooling_cnt=4, fit_num=8):
    img_hei, img_wid = img.shape
    fit_num = 2**pooling_cnt
    img = img[img_hei%fit_num:, img_wid%fit_num:]
    return  img

class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, label_dir, transform=None, path_expend=True):

        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform

        lst_img = os.listdir(self.img_dir)
        lst_label = os.listdir(self.label_dir)

        self.path_expend = path_expend

        lst_mask = [f for f in lst_label if f.startswith('PANO')]
        lst_mask.sort()
        self.lst_mask = lst_mask

    def __len__(self):
        return  len(self.lst_mask)

    def __getitem__(self, idx):
        json_path = self.label_dir + self.lst_mask[idx]

        with open(json_path, 'r') as jsonfile:
            jsondata = json.load(jsonfile)
            if self.path_expend == True:
                image = cv2.imread(self.img_dir+jsondata["filename"])
            elif self.path_expend == False:
                image = cv2.imread(jsondata["filename"])

            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            img_hei, img_wid = image.shape

            croped_image = fit_8(image, pooling_cnt=4, fit_num=8)

            bg_mask = np.zeros(image.shape, dtype=np.uint8)
            contours = []

            for Caries in jsondata["cariesCoords"]:
                caries_contours = []
                for Caries_xy in Caries:
                    x = Caries_xy["x"]
                    y = Caries_xy["y"]
                    caries_contours.append([[x,y]])

                caries_contours = np.array(caries_contours, dtype=np.int16)
                ctr = np.array(caries_contours).reshape((-1, 1, 2)).astype(np.int32)
                cv2.drawContours(bg_mask, [ctr], -1, 255, -1)
            croped_bg = fit_8(bg_mask, pooling_cnt=4, fit_num=8)

        image = croped_image / 255.0
        mask = croped_bg / 255.0

        if image.ndim == 2:
            image = image[:,:, np.newaxis]
        if mask.ndim == 2:
            mask = mask[:,:, np.newaxis]

        data = {'image' : image, 'mask' : mask}

        if self.transform:
            data = self.transform(data)

        return data

class ToTensor(object):
    def __call__(self, data):
        image, mask = data['image'], data['mask']
        mask = mask.transpose((2, 0, 1)).astype(np.float32)
        image = image.transpose((2, 0, 1)).astype(np.float32)

        data = {'image' : torch.from_numpy(image), 'mask' : torch.from_numpy(mask)}
        return data

class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        image, mask = data['image'], data['mask']

        image = (image - self.mean) / self.std
        data = {'image' : image, 'mask' : mask}
        return data

transform = transforms.Compose([ToTensor()])

dataset_train = Dataset(image_train_work_dir, label_train_work_dir, transform=transform, path_expend=True)
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=2)

dataset_val = Dataset(image_vaild_work_dir, label_valid_work_dir, transform=transform, path_expend=False)
loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=Flase, num_workers=2)

net = UNet().to(device)

if torch.cuda.device_count() > 1:
    net = nn.DataParallel(net, device_ids=[0,1])

fn_loss = nn.BCEWithLogitsLoss().to(device)

optim = torch.optim.Adam(net.parameters(), lr=lr)
scheduler = StepLR(optim, step_size=1, gamma=0.997)

num_data_train = len(dataset_train)
num_data_val = len(dataset_val)
num_batch_train = np.ceil(num_data_train / batch_size)
num_batch_val = np.ceil(num_data_val / batch_size)

fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0,2,3,1)
fn_denorm = lambda x, mean, std: (x * std) + mean
fn_class = lambda x: 1.0 * (x > 0.5)

def save(ckpt_dir, net, optim, epoch):
    createFolder(ckpt_dir)

    torch.save({'net':net.state_dict(), 'optim':optim.state_dict()}, "%s/model_epoch%d.pth"%(ckpt_dir, epoch))

def load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return  net, optim, epoch

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('%s%s' % (ckpt_dir, ckpt_lst[-1]))

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return net, optim, epoch

st_epoch = 0

if retrain == True:
    net, optim, st_epoch = load(ckpt_dir = retrain_ckpt_dir, net=net, optim=optim)
    print("ReTrain start !!!")

load_idx = 0
min_loss = 1

for epoch in range(st_epoch + 1, num_epoch + 1):
    net.train()
    loss_arr=[]
    print(' Epoch : ', epoch,' LR : ', scheduler.get_lr(), ' ', train_name)

    pbar = tqdm(total = len(loader_train))
    for batch, data in enumerate(loader_train, 1):
        mask = data['mask'].to(device)
        image = data['image'].to(device)

        output = net(image)
        optim.zero_grad()
        loss = fn_loss(output, mask)
        loss.backward()
        optim.step()

        loss_arr += [loss.item()]

        postfix = OrderedDict([
            ('loss', np.mean(loss_arr))
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    with torch.no_grad():
        net.eval()
        loss_arr = []

        pbar = tqdm(total = len(loader_val))
        for batch, data in enumerate(loader_val, 1):
            mask = data['mask'].to(device)
            image = data['image'].to(device)

            output = net(image)

            loss = fn_loss(output, mask)
            loss_arr += [loss.item()]

            postfix = OrderedDict([
                ('loss', np.mean(loss_arr))
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    if epoch % 5 == 0:
        save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)
        min_loss = np.mean(loss_arr)
        print("!!!! Step Save !!!!")
    scheduler.step()

