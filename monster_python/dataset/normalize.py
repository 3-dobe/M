import re

from PIL import Image

from utils import *
import os


TRAIN_DIR = "dataset/train/"
TRAIN_NORM_DIR = TRAIN_DIR + "norm/"
TRAIN_MATERIAL_DIR = TRAIN_DIR + "material/"
TRAIN_ORIGIN_DIR = TRAIN_DIR + "origin/"

TRAIN_DATA_INIT = TRAIN_DIR + "haar_init.data"
TRAIN_MODEL = TRAIN_DIR + "haar_classifier.model"


def __init_origin(material_dir, origin_dir, tween=None, start=0):
    for f in os.listdir(material_dir):
        f_g = re.split("[_.]", f)
        if len(f_g) == 3 and (f_g[0] == "p" or f_g[0] == "n") and f_g[2] == "png" and int(f_g[1]) >= 0:
            n_img = ImgUtil.complete_scale_to_img(material_dir + f, 0, tween)
            n_img.save(origin_dir + f_g[0] + "_" + str(start + int(f_g[1])) + "." + f_g[2], "png")


def __normalize(origin_dir, train_dir, resample=Image.NEAREST):
    for f in os.listdir(origin_dir):
        f_g = re.split("[_.]", f)
        if len(f_g) == 3 and (f_g[0] == "p" or f_g[0] == "n") and f_g[2] == "png" and int(f_g[1]) >= 0:
            n_img = ImgUtil.complete_scale_to_img(origin_dir + f, 24, None, resample)
            n_img.save(train_dir + "norm_" + f, "png")


def normalize_big_n(big_img_path, crop_s):
    train = "train/"
    img_list = ImgUtil.crop_img_list(big_img_path, crop_s)
    for i in range(len(img_list)):
        img_resized = img_list[i].resize((24, 24))
        img_resized.save(train + "norm_n_" + str(i) + ".png", "png")


if __name__ == '__main__':
    __init_origin(TRAIN_MATERIAL_DIR, TRAIN_ORIGIN_DIR, TRAIN_MATERIAL_DIR + "tween_1.png")
    __normalize(TRAIN_ORIGIN_DIR, TRAIN_NORM_DIR, Image.ANTIALIAS)
    # normalize_big_n("train/origin/big_n_2.png", 60)
