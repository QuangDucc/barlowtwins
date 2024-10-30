import pandas as pd
import numpy as np
import nibabel as nib
import torch
import torchio as tio
from typing import Iterator, List, Optional, Union
from torch.utils.data import DistributedSampler
from torch.utils.data.sampler import Sampler
from catalyst.data.dataset import DatasetFromSampler
from operator import itemgetter
from utils.training import data
import glob
from operator import itemgetter
import pydicom as dicom
import SimpleITK as sitk


def thru_plane_position(dcm):
    """Gets spatial coordinate of image origin whose axis
    is perpendicular to image plane.
    """
    orientation = tuple((float(o) for o in dcm.ImageOrientationPatient))
    position = tuple((float(p) for p in dcm.ImagePositionPatient))
    rowvec, colvec = orientation[:3], orientation[3:]
    normal_vector = np.cross(rowvec, colvec)
    slice_pos = np.dot(position, normal_vector)
    return slice_pos


def read_vol(row, augs, crop_size):

    dicom_folder = row['phase_dir']
    list_dicom = glob.glob(dicom_folder+"/*")
    dcm_slices = []
    for file in list_dicom: 
        if "version" in file.lower():
            continue
        dcm_slices.append((file, thru_plane_position(dicom.read_file(file))))

    dcm_slices = sorted(dcm_slices, key=itemgetter(1))
    
    list_position_sorted = [i[0] for i in dcm_slices]
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(tuple(list_position_sorted))
    data_itk = reader.Execute()
    vol = sitk.GetArrayFromImage(data_itk)
    
    # crop or pad
    vol = data.volume_utils.crop_or_pad(vol, 1, 320)
    vol = data.volume_utils.crop_or_pad(vol, 2, 320)

    if vol.shape[0] < crop_size[0]:
        vol = data.volume_utils.crop_or_pad(vol, 0, crop_size[0])
    
    vol = torch.tensor(vol[np.newaxis], dtype=torch.float).clip(-100, 300)
    original_subject = tio.Subject(volume=tio.ScalarImage(tensor=vol))
    
    # create patches=2
    patch_sampler = tio.data.UniformSampler(crop_size)
    patches = []
    for p in patch_sampler(original_subject):
        if len(patches) == 2:
            break

        z0, x0, y0, z1, x1, y1 = p[tio.LOCATION]
        patches.append(vol[:, z0:z1, x0:x1, y0:y1])
    
    patch_subjects = [tio.Subject(volume=tio.ScalarImage(tensor=p)) for p in patches]
    aug_patch_subjects_1 = [augs(p).volume.data for p in patch_subjects]
    aug_patch_subjects_2 = [augs(p).volume.data for p in patch_subjects]
    
    patches_1 = torch.cat(aug_patch_subjects_1, dim=0)
    patches_2 = torch.cat(aug_patch_subjects_2, dim=0)
    
    # intance normalization
    std1, mean1 = torch.std_mean(patches_1, dim=[-1, -2, -3], keepdim=True)
    std2, mean2 = torch.std_mean(patches_2, dim=[-1, -2, -3], keepdim=True)
    
    x1 = (patches_1 - mean1) / std1
    x2 = (patches_2 - mean2) / std2
    
    # possibly cause overflow -> clip
    x1 = x1.clip(-10, 10)
    x2 = x2.clip(-10, 10)

    return x1, x2 


def train(df_path='../ct_total.csv', crop_size=[64, 144, 144], return_ds=False, **kwargs):
    df = pd.read_csv(df_path)
    
    augs = tio.transforms.Compose([
        tio.transforms.RandomFlip(axes=[0], flip_probability=0.5),
        tio.transforms.RandomAffine(scales=0.01, translation=0, degrees=(10, 0, 0), p=0.5),
        tio.transforms.RandomElasticDeformation(num_control_points=5, max_displacement=12, p=0.5),
        tio.OneOf({
            # tio.transforms.RandomBiasField(coefficients=1): 1,
            tio.transforms.RandomGamma(log_gamma=0.18): 1,
        }, p=0.8),
        tio.transforms.RandomBlur(std=(0, 1.1), p=0.5),
        tio.transforms.RandomNoise(std=(0, 50), p=0.5),
    ])
    
    ds = data.PandasDataset(df, read_vol, augs=augs, crop_size=crop_size)
    
    if return_ds:
        return ds
    
    return ds.get_torch_loader(**kwargs)


class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.

    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.

    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        """

        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
                distributed training
            rank (int, optional): Rank of the current process
                within ``num_replicas``
            shuffle (bool, optional): If true (default),
                sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        """Iterate over sampler.

        Returns:
            python iterator
        """
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))