from haar_adaboost.haar import Haar
from utils import *


if __name__ == '__main__':
    img_path = "E:\\workspace\\chaopei\\m\\monster_python\\dataset\\m_1.jpg"
    img = ImgUtil.crop_scale_to_array(img_path, 24)
    haar = Haar(img)
    print("test_haar: haar[5]=" + str(haar[5]))
