import cv2
import os
from tqdm import tqdm
from glob import glob
from fire import Fire
from predict import Predictor
# from skimage.measure import compare_ssim as SSIM
from skimage.metrics import structural_similarity as SSIM
from util.metrics import PSNR
import numpy as np


class Evaler:
    def __init__(self):
        pass

    def get_metrics(self, out, tar):
        psnr = PSNR(out, tar)
        ssim = SSIM(out, tar, multichannel=True)
        return psnr, ssim


def main(img_pattern,
         mask_pattern,
         weights_path,
         out_dir=None):
    def sorted_glob(pattern):
        return sorted(glob(pattern))

    imgs = sorted_glob(img_pattern)
    masks = sorted_glob(mask_pattern) if mask_pattern is not None else [None for _ in imgs]
    pairs = zip(imgs, masks)
    names = sorted([os.path.basename(x) for x in glob(img_pattern)])
    predictor = Predictor(weights_path=weights_path)
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
    evaler = Evaler()
    ssims = []
    psnrs = []
    for name, pair in tqdm(zip(names, pairs), total=len(names)):
        f_img, f_mask = pair
        img, mask = map(cv2.imread, (f_img, f_mask))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        pred = predictor(img, mask)
        metrics = evaler.get_metrics(pred, mask)
        ssims.append(metrics[1])
        psnrs.append(metrics[0])
        # print("\nbatch_id:", name, "PSNR:", metrics[0], "SSIM:", metrics[1])
        if out_dir is not None:
            pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(out_dir, name),
                        pred)
    print("SSIM:", np.mean(ssims), "PSNR:", np.mean(psnrs))
    exit(0)


if __name__ == '__main__':
    Fire(main)