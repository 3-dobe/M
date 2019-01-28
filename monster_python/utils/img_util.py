import numpy as np
from PIL import Image


class ImgUtil:
    @staticmethod
    def crop_scale_to_array(img_path, s):
        """确保输入的图片尺寸是sxs的, 转为灰度图, 若图片原尺寸不是正方形, 需要先裁剪后缩放

        :param img_path: 输入的图片地址
        :param s: 尺寸边长
        :return: 图像数据数组
        """
        return np.array(ImgUtil.crop_scale_to_img(img_path, s))

    @staticmethod
    def crop_scale_to_img(img_path, s):
        """确保输入的图片尺寸是sxs的, 转为灰度图, 若图片原尺寸不是正方形, 需要先裁剪后缩放

        :param img_path: 输入的图片地址
        :param s: 尺寸边长
        :return: 图像Image对象
        """
        im = Image.open(img_path, "r").convert("L")
        w, h = im.size
        if w != h:
            if w > h:
                box = (0, 0, h, h)
            else:
                box = (0, 0, w, w)
            im = im.crop(box)
        return im.resize((s, s), Image.ANTIALIAS)
