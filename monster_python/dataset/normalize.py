from utils import *
import os


def normalize():
    origin = "train/origin/"
    train = "train/"
    for f in os.listdir(origin):
        n_img = ImgUtil.complete_scale_to_img(origin + f, 24)
        n_img.save(train + "norm_" + f, "png")


def normalize_big_n(big_img_path, crop_s):
    train = "train/"
    img_list = ImgUtil.crop_img_list(big_img_path, crop_s)
    for i in range(len(img_list)):
        img_resized = img_list[i].resize((24, 24))
        img_resized.save(train + "norm_n_" + str(i) + ".png", "png")


if __name__ == '__main__':
    normalize()
    # normalize_big_n("train/origin/big_n_2.png", 60)
