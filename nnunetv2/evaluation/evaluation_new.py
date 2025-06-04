from nnunetv2.paths import nnUNet_results, nnUNet_raw
from batchgenerators.utilities.file_and_folder_operations import (
    load_json, join, isdir, listdir, save_json
)
import nibabel as nib
import torch

class Evaluator:
    def __init__(self, )