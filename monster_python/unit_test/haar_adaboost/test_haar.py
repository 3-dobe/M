import numpy as np

from PIL import Image

from haar_adaboost.haar import Haar
from utils import *


if __name__ == '__main__':
    test_img_path = "E:\\workspace\\chaopei\\M\\monster_python\\unit_test\\haar_adaboost\\test_img.png"
    test_img = np.array(Image.open(test_img_path, "r").convert("L")) # [[row] [] []]
    test_img_ig = Haar.obtain_integral_graph(test_img)

    p_1 = "E:\\workspace\\chaopei\\M\\monster_python\\unit_test\\haar_adaboost\\p_1.png"
    # 24*24
    img_n = ImgUtil.complete_scale_to_array(p_1, 24)
    # img_n.save(n_p_1, "png")
    haar_n_p_1 = Haar(p_1, img_n)
    print(haar_n_p_1[0:100])
    # 48*48
    img_n2 = ImgUtil.complete_scale_to_array(p_1, 48)
    # img_n2.save(n2_p_1, "png")
    haar_n2_p_1 = Haar(p_1, img_n2, 2)
    print(haar_n2_p_1[0:100])
    print("ok")

