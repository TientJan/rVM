# python imports
import os
import glob
# external imports
import torch
import numpy as np

import SimpleITK as sitk
# internal imports
from Model import losses
from Model.config import args
from Model.model import U_Network, SpatialTransformer
from Model.datagenerators_affine import Dataset
import torch.utils.data as Data

import time
from datetime import timedelta

os.environ["CUDA_VISIBLE_DEVICES"]="4"

def make_dirs():
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)


def save_image(img, ref_img, name):
    img = sitk.GetImageFromArray(img[0, 0, ...].cpu().detach().numpy())
    ref_img = sitk.GetImageFromArray(ref_img[0, 0, ...].cpu().detach().numpy())
    img.SetOrigin(ref_img.GetOrigin())
    img.SetDirection(ref_img.GetDirection())
    img.SetSpacing(ref_img.GetSpacing())
    sitk.WriteImage(img, os.path.join(args.result_dir, name))


def compute_label_dice(gt, pred):
    # 需要计算的标签类别，不包括背景和图像中不存在的区域
    cls_lst = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 61, 62,
               63, 64, 65, 66, 67, 68, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 101, 102, 121, 122, 161, 162,
               163, 164, 165, 166]
    dice_lst = []
    for cls in cls_lst:
        dice = losses.DSC(gt == cls, pred == cls)
        dice_lst.append(dice)
    return np.mean(dice_lst)


# @torchsnooper.snoop()
def test():
    make_dirs()
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    print(args.checkpoint_path)

    # f_img = sitk.ReadImage(args.atlas_file)
    # input_fixed = sitk.GetArrayFromImage(f_img)[np.newaxis, np.newaxis, ...]
    vol_size = (160, 192, 160)
    # # set up atlas tensor
    # input_fixed = torch.from_numpy(input_fixed).to(device).float()

    # Test file and anatomical labels we want to evaluate
    DS = Dataset(["../../Dataset/RESECT/resize_affine/test/*/*T1.nii",
                  "../../Dataset/RESECT/resize_affine/test/*/*FLAIR.nii"])

    print("The number of test data: ", len(DS))

    DL = Data.DataLoader(DS, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)
    # Prepare the vm1 or vm2 model and send to device
    nf_enc = [16, 32, 32, 32]
    if args.model == "vm1":
        nf_dec = [32, 32, 32, 32, 8, 8]
    else:
        nf_dec = [32, 32, 32, 32, 32, 16, 16]
    # Set up model
    UNet = U_Network(len(vol_size), nf_enc, nf_dec).to(device)
    UNet.load_state_dict(torch.load(args.checkpoint_path))
    STN_img = SpatialTransformer(vol_size).to(device)
    STN_label = SpatialTransformer(vol_size, mode = "nearest").to(device)
    UNet.eval()
    STN_img.eval()
    STN_label.eval()

    DSC = []

    mod1 = slice(0, 160)
    mod2 = slice(160, 320)
    # fixed图像对应的label
    # fixed_label = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(args.label_dir, "S01.delineation.structure.label.nii.gz")))
    for i, data in enumerate(DL):
        start_time = time.time()
        name = DS.file_id[i]
        # 读入moving图像
        input_moving = data[:,:,:,:,mod1].to(device).float()
        input_fixed = data[:,:,:,:,mod2].to(device).float()
        # 读入moving图像对应的label
        # label_file = glob.glob(os.path.join(args.label_dir, name[:3] + "*"))[0]
        # input_label = sitk.GetArrayFromImage(sitk.ReadImage(label_file))[np.newaxis, np.newaxis, ...]
        # input_label = torch.from_numpy(input_label).to(device).float()

        # 获得配准后的图像和label
        pred_flow = UNet(input_moving, input_fixed)
        pred_img = STN_img(input_moving, pred_flow)
        # pred_label = STN_label(input_label, pred_flow)

        # # 计算DSC
        # dice = compute_label_dice(fixed_label, pred_label[0, 0, ...].cpu().detach().numpy())
        # print("dice: ", dice)
        # DSC.append(dice)

        save_image(pred_img, input_fixed, "{}_warped.nii.gz".format(name))
        save_image(pred_flow.permute(0, 2, 3, 4, 1)[np.newaxis, ...], input_fixed, "{}_flow.nii.gz".format(name))
        # save_image(pred_label, f_img, "7_label.nii.gz")
        # del pred_flow, pred_img, pred_label
        excute_time = time.time() - start_time

        print("Excute time is %s sec" % timedelta(seconds = round(excute_time)) )

    print("mean(DSC): ", np.mean(DSC), "   std(DSC): ", np.std(DSC))    


if __name__ == "__main__":
    test()
