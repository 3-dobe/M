import numpy as np
from PIL import Image


class ImgUtil:
    @staticmethod
    def complete_scale_to_array(img_path, s):
        """确保输出的图片尺寸是sxs的, 转为灰度图, 若图片原尺寸不是正方形, 需要先补全后缩放

        :param img_path: 输入的图片地址
        :param s: 尺寸边长
        :return: 图像数据数组
        """
        return np.array(ImgUtil.complete_scale_to_img(img_path, s))

    @staticmethod
    def complete_scale_to_img(img_path, s):
        """确保输出的图片尺寸是sxs的, 转为灰度图, 若图片原尺寸不是正方形, 需要先补全后缩放

        :param img_path: 输入的图片地址
        :param s: 尺寸边长
        :return: 图像Image对象
        """
        im = Image.open(img_path, "r").convert("L")  # 灰度
        w, h = im.size
        if w != h:
            if w > h:
                box = (0, 0, w, w)
            else:
                box = (0, 0, h, h)
            im = im.crop(box)
        return im.resize((s, s))

    @staticmethod
    def crop_img_list(img_path, s):
        """确保输入的图片列表中所有图片尺寸都是sxs的, 转为灰度图, 若图片原尺寸不是正方形, 需要先补全后缩放

        :param img_path: 输入的原大图片地址
        :param s: 尺寸边长
        :return: 图像Image对象列表
        """
        im_list = []
        origin_img = Image.open(img_path, "r").convert("L")  # 灰度
        orig_w, orig_h = origin_img.size
        for x in range(0, orig_w, s):
            for y in range(0, orig_h, s):
                box = (x, y, x + s, y + s)
                im = origin_img.crop(box)
                im_list.append(im)
        return im_list
