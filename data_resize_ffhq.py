import multiprocessing
from functools import partial
from io import BytesIO
from pathlib import Path

import lmdb
import glob
from PIL import Image
from torchvision.transforms import functional as trans_fn
from tqdm import tqdm
import os


def resize_and_convert(img, size, resample, quality=100):
    img = trans_fn.resize(img, size, resample)
    img = trans_fn.center_crop(img, size)
    buffer = BytesIO()
    img.save(buffer, format="png", quality=quality)
    val = buffer.getvalue()

    return val


def resize_multiple(img,
                    sizes=(128, 256, 512, 1024),
                    resample=Image.LANCZOS,
                    quality=100):
    imgs = []

    for size in sizes:
        imgs.append(resize_and_convert(img, size, resample, quality))

    return imgs


def resize_worker(img_file, sizes, resample):
    i, (file, idx) = img_file
    img = Image.open(file)
    img = img.convert("RGB")
    out = resize_multiple(img, sizes=sizes, resample=resample)

    return i, idx, out


def prepare(env,
            paths,
            n_worker,
            sizes=(128, 256, 512, 1024),
            resample=Image.LANCZOS):
    resize_fn = partial(resize_worker, sizes=sizes, resample=resample)

    # index = filename in int
    indexs = []
    for each in paths:
        file = os.path.basename(each)
        name, ext = file.split('.')
        idx = int(name)
        indexs.append(idx)

    # sort by file index
    files = sorted(zip(paths, indexs), key=lambda x: x[1])
    files = list(enumerate(files))
    total = 0

    with multiprocessing.Pool(n_worker) as pool:
        for i, idx, imgs in tqdm(pool.imap_unordered(resize_fn, files)):
            for size, img in zip(sizes, imgs):
                key = f"{size}-{str(idx).zfill(5)}".encode("utf-8")

                with env.begin(write=True) as txn:
                    txn.put(key, img)

            total += 1

        with env.begin(write=True) as txn:
            txn.put("length".encode("utf-8"), str(total).encode("utf-8"))


if __name__ == "__main__":
    """
    converting ffhq images to lmdb
    """
    num_workers = 1
    # original ffhq data path
    in_path = '../CheXpert_Dataset/images_256/images'
    # target output path
    out_path = '../CheXpert_Dataset/images_256/chexpert.lmdb'

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    resample_map = {"lanczos": Image.LANCZOS, "bilinear": Image.BILINEAR}
    resample = resample_map['lanczos']

    sizes = [256]
    print(f"Make dataset of image sizes:", ", ".join(str(s) for s in sizes))

    exts = ['png']
    paths = [p for ext in exts for p in Path(f'{in_path}').glob(f'*.{ext}')]
    print(paths[0])
    # paths = glob.glob(os.path.join(in_path, '*.png'))

    with lmdb.open(out_path, map_size=1024**4, readahead=False) as env:
        prepare(env, paths, num_workers, sizes=sizes, resample=resample)
