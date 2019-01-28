from utils import ImgUtil


def test_crop_scale_to_array():
    img_path = "E:\\workspace\\chaopei\\m\\monster_python\\dataset\\m_1.jpg"
    array = ImgUtil.crop_scale_to_array(img_path, 28)
    print(array)


def test_crop_scale_to_img():
    img_path = "E:\\workspace\\chaopei\\m\\monster_python\\dataset\\m_1.jpg"
    img_path_to = "E:\\workspace\\chaopei\\m\\monster_python\\dataset\\m_1_to.png"
    ImgUtil.crop_scale_to_img(img_path, 28).save(img_path_to, "png")


if __name__ == '__main__':
    test_crop_scale_to_img()
    test_crop_scale_to_array()
