import argparse
from io import BytesIO
import multiprocessing
from functools import partial
import os
import glob
from PIL import Image
import pickle
import lmdb
import numpy as np
from tqdm import tqdm
from torchvision import datasets
from torchvision.transforms import functional as trans_fn
from skimage.transform import resize


def resize_brats(img, size):
    img = resize(img, (size, size*2))

    return img


def resize_multiple_brats(save_dir, img_file, img, sizes=(8, 16, 32, 64, 128, 256, 512, 1024), quality=100):
    imgs = []

    for size in sizes:
        resized_img = resize_brats(img, size)

        with open(os.path.join(save_dir, str(size), img_file[1].split('/')[-1]),'wb') as f:
            pickle.dump(resized_img, f)
    return imgs


def resize_worker_brats(img_files, save_dir, sizes):
    max_list = [6476.0, 6655.0, 8979.0, 3918.0]
    for size in sizes:
        os.makedirs(os.path.join(save_dir, str(size)),exist_ok=True)
    for img_file in img_files:
        i, file = img_file
        with open(file, 'rb') as f:
            pickle_data = pickle.load(f)

        # z-score normalization
        image = pickle_data['image']
        image = np.pad(image, ((4, 4), (8, 8), (0, 0)), 'constant')
        for j in range(4):
            image[:, :, j] = ((image[:, :, j] / max_list[j]))

        out = resize_multiple_brats(save_dir, img_file, image, sizes=sizes)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str)
    parser.add_argument('--n_worker', type=int, default=8)
    #parser.add_argument('path', type=str)

    args = parser.parse_args()
    sizes = (8, 16, 32, 64, 128)
    resize_fn = partial(resize_worker_brats, save_dir = args.out, sizes=sizes)
    data_dir = '/home/nas1_temp/minsoolee/tumor/dataset/'
    pickle_path = os.path.join(data_dir, 'tumor')
    pickle_list = glob.glob(os.path.join(pickle_path, '*'))
    files = [(i, file) for i, file in enumerate(pickle_list)]

    resize_worker_brats(files, args.out, sizes=sizes)

    # with multiprocessing.Pool(args.n_worker) as pool:
    #    for i, _ in pool.imap_unordered(resize_fn, files):
    #        print(i)

