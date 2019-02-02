from multiprocessing import Process

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
        return WeakClassifier(s["feature_index"], s["feature_sep"], Symbol.LtEq, s["err"], s["alpha"])
    elif s["symbol"] == Symbol.Gt.value:
        return WeakClassifier(s["feature_index"], s["feature_sep"], Symbol.Gt, s["err"], s["alpha"])
    else:
        return None


def __read_train_model(model_path):
    return get_cache_obj(model_path, decoder=__json_decode_classifier)


if __name__ == '__main__':
    # read model
    strong_classifier = __read_train_model(ROOT_DIR + normalize.TRAIN_MODEL)
    if strong_classifier is not None:
        total_alpha = 0
        for w_c in strong_classifier:
            total_alpha += w_c.alpha
            Log.d("test_adaboost: %s" % (str(w_c)))
        # test-img norm
        test_img_name = "test_1.jpg"
        test_img_path = ROOT_DIR + normalize.TEST_DIR + test_img_name
        test_img = Image.open(test_img_path, "r").convert("L")
        w, h = test_img.size
        if w >= h:
            max_s = w
        else:
            max_s = h

        Log.d("test_adaboost: BEGIN")
        mult = 0
        for crop_s in range(normalize.HAAR_SIZE, max_s + 1, normalize.HAAR_SIZE):
            mult += 1
            for x in range(0, w, mult * 2):
                for y in range(0, h, mult * 2):
                    crop_img_name = "mult%d_x%d_y%d_s%d_%s" % (mult, x, y, crop_s, test_img_name)
                    crop_img = test_img.crop((x, y, x + crop_s, y + crop_s))
                    crop_haar = Haar(crop_img_name, np.array(crop_img), mult)
                    # test
                    h_alpha = 0
                    for w_c in strong_classifier:
                        if (w_c.symbol == Symbol.Gt and crop_haar[w_c.feature_index] > w_c.feature_sep) or \
                                (w_c.symbol == Symbol.LtEq and crop_haar[w_c.feature_index] <= w_c.feature_sep):
                            h_alpha += w_c.alpha
                    Log.d("test_adaboost: detecting %s, h_alpha=%f, total_alpha=%f" %
                          (crop_img_name, h_alpha, total_alpha))
                    if h_alpha >= total_alpha * 0.85:
                        Log.e("test_adaboost: detected")
                        crop_img.save(ROOT_DIR + normalize.TEST_NORM_DIR + crop_img_name, "png")
                    else:
                        Log.d("test_adaboost: undetected")
        Log.d("test_adaboost: END")
    else:
        Log.e("test_adaboost: no model read")
