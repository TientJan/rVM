import nibabel as nib
import glob
import scipy.ndimage
import numpy as np
import os

def read(path):
    return nib.load(path).get_fdata()

def resize(img, shape):
    factors = (
        shape[0] / img.shape[0],
        shape[1] / img.shape[1],
        shape[2] / img.shape[2],
    )
    return scipy.ndimage.zoom(img, factors, mode = "constant")
"""============== RESECT ================="""
# def crop(img):
# 	_, y, z = img.shape
# 	img = img[:,8:y-8, 48:z-16]
# 	return img
out_root = "../../Dataset/RESECT/resize/train"
paths = glob.glob("../../Dataset/RESECT/raw/train/*/MRI/*.nii.gz")


for path in paths:
    # RESECT
    # img = crop(read(path)) 
    img = read(path)
  
    # name = os.path.splitext(os.path.basename(path))
    npath = path.replace("raw","resize")
    npath = npath.replace("/MRI","")
    # ndir: "......train/Case1"
    ndir = os.path.split(npath)[0]
    if not os.path.exists(ndir):
        os.makedirs(ndir)
     # npath: "......train/Case1/Case1-T1"
    npath = npath.replace(".nii.gz","")
    print(npath)
    img = resize(img, (160, 192, 160))
    # nib.save(nib.Nifti1Image(img, np.eye(4)), os.path.join(out_root, name))
    nib.save(nib.Nifti1Image(img, np.eye(4)), npath)
"""========================"""

# out_root = "../../Dataset/jzl/resize/train"
# path1s = glob.glob("../../Dataset/jzl/raw/train/*/*t1*.nii")
# path2s = glob.glob("../../Dataset/jzl/raw/train/*/*t2*.nii")

# for path in path1s:
    
#     img = read(path)[::-1, 16:, :]
#     # name = os.path.splitext(os.path.basename(path))
#     npath = path.replace("raw","resize")
#     # ndir: "......train/2020...XU_FUN"
#     ndir = os.path.split(npath)[0]
#     if not os.path.exists(ndir):
#         os.mkdir(ndir)
#      # npath: "......train/Case1/Case1-T1"
#     npath = npath.replace(".nii","")
#     print(npath)
#     img = resize(img, (192, 240, 16))
#     # nib.save(nib.Nifti1Image(img, np.eye(4)), os.path.join(out_root, name))
#     nib.save(nib.Nifti1Image(img, np.eye(4)), npath)


# for path in path2s:
    
#     img = read(path)[:, 16:, :]
#     # name = os.path.splitext(os.path.basename(path))
#     npath = path.replace("raw","resize")
#      # npath: "......train/Case1/Case1-T1"
#     npath = npath.replace(".nii","")
#     print(npath)
#     img = resize(img, (192, 240, 16))
#     # nib.save(nib.Nifti1Image(img, np.eye(4)), os.path.join(out_root, name))
#     nib.save(nib.Nifti1Image(img, np.eye(4)), npath)