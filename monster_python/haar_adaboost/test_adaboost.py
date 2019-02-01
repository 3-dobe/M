import numpy as np

from PIL import Image

import dataset.normalize as normalize
from haar_adaboost import WeakClassifier, Haar
from haar_adaboost import Symbol
from utils import Log
from utils.cache_obj import get_cache_obj

ROOT_DIR = "E:/workspace/chaopei/"


def __json_decode_classifier(s):
    if s["symbol"] == Symbol.LtEq.value:
        return WeakClassifier(s["feature_index"], s["feature_index"], Symbol.LtEq, s["err"], s["alpha"])
    elif s["symbol"] == Symbol.Gt.value:
        return WeakClassifier(s["feature_index"], s["feature_index"], Symbol.Gt, s["err"], s["alpha"])
    else:
        return None


def __read_train_model(model_path):
    return get_cache_obj(model_path, decoder=__json_decode_classifier)


if __name__ == '__main__':
    # read model
    strong_classifier = __read_train_model(ROOT_DIR + normalize.TRAIN_MODEL)
    if strong_classifier is not None:
        for weak_classifier in strong_classifier:
            Log.d("test_adaboost: %s" % (str(weak_classifier)))
    # test-img norm
    test_img_name = "test_1.jpg"
    test_img_path = ROOT_DIR + normalize.TEST_DIR + test_img_name
    test_img = Image.open(test_img_path, "r")
    w, h = test_img.size
    if w >= h:
        max_s = w
    else:
        max_s = h

    Log.d("test_adaboost: BEGIN")
    mult = 0
    for crop_s in range(normalize.HAAR_SIZE, max_s + 1, normalize.HAAR_SIZE):
        mult += 1
        if mult != 3:
            continue
        for x in range(0, w, mult * 2):
            for y in range(0, h, mult * 2):
                crop_img_name = "mult%d_x%d_y%d_s%d_%s" % (mult, x, y, crop_s, test_img_name)
                crop_img = test_img.crop((x, y, x + crop_s, y + crop_s))
                # todo save?
                crop_img.save(ROOT_DIR + normalize.TEST_NORM_DIR + crop_img_name, "png")
                # crop_haar = Haar(crop_img_name, crop_img, mult)
    Log.d("test_adaboost: END")
