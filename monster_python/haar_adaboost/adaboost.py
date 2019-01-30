import json
from enum import Enum

import numpy as np
import os
import re

from PIL import Image

from haar_adaboost import Haar


class Symbol(Enum):
    Gt = 1
    LtEq = 0


class PN(Enum):
    Positive = 1
    Negative = 0


class WeakClassifier:
    def __init__(self, feature_index, feature_sep, symbol, err, alpha=0):
        self.feature_index = feature_index
        self.feature_sep = feature_sep
        self.symbol = symbol
        self.err = err
        self.alpha = alpha

    def __str__(self):
        return "WeakClassifier: feature_index=" + str(self.feature_index) + \
               ", feature_index=" + str(self.feature_index) + \
               ", symbol=" + str(self.symbol) + \
               ", err=" + str(self.err) + \
               ", alpha=" + str(self.alpha)


def __json_encode_classifier(s):
    if isinstance(s, WeakClassifier):
        return {
            "feature_index": s.feature_index,
            "feature_sep": s.feature_sep,
            "symbol": s.symbol,
            "err": s.err,
            "alpha": s.alpha
        }
    elif isinstance(s, Symbol):
        return s.value


def __json_decode_classifier(s):
    if s["symbol"] == Symbol.LtEq.value:
        return WeakClassifier(s["feature_index"], s["feature_index"], Symbol.LtEq, s["err"], s["alpha"])
    elif s["symbol"] == Symbol.Gt.value:
        return WeakClassifier(s["feature_index"], s["feature_index"], Symbol.Gt, s["err"], s["alpha"])
    else:
        return None


class SampleFeature:
    def __init__(self, sample, f_i):
        self.sample = sample
        self.feature_index = f_i
        self.feature = self.sample.features[f_i]
        self.pn = self.sample.pn
        self.weight = self.sample.weight


class Sample:
    def __init__(self, filename, features, pn, weight):
        self.filename = filename
        self.features = features
        self.pn = pn
        self.weight = weight


def __json_encode_sample(s):
    if isinstance(s, Sample):
        return {"filename": s.filename, "features": s.features, "pn": s.pn, "weight": s.weight}
    elif isinstance(s, PN):
        return s.value


def __json_decode_sample(s):
    if s["pn"] == PN.Positive.value:
        return Sample(s["filename"], s["features"], PN.Positive, s["weight"])
    elif s["pn"] == PN.Negative.value:
        return Sample(s["filename"], s["features"], PN.Negative, s["weight"])
    else:
        return None


def __cache_obj(cache_path, obj, encoder):
    print("__cache_obj: cache_path=" + cache_path)
    json_str = json.dumps(obj, default=encoder)
    cache = open(cache_path, "w")
    cache.write(json_str)
    cache.close()


def __get_cache_obj(cache_path, decoder):
    print("__get_cache_obj: cache_path=" + cache_path)
    cache = None
    try:
        cache = open(cache_path, "r")
        json_str = cache.read()
        return json.loads(json_str, object_hook=decoder)
    except:
        return None
    finally:
        if cache is not None:
            cache.close()


def __haar_index(f):
    """['norm', 'p', '1', 'png]

    """
    return re.split("[_.]", f)


def __init_current_haars_with_weight():
    cache_path = "../dataset/train/haar_init.data"
    haars = __get_cache_obj(cache_path, __json_decode_sample)
    if haars is None:
        train_data_dir = "../dataset/train/"
        # tuple list
        haars = []
        for f in os.listdir(train_data_dir):
            if f.startswith("norm_") and f.endswith(".png"):
                haar = Haar(f, np.array(Image.open(train_data_dir + f, "r")))
                haar.cal()
                hi = __haar_index(f)
                if hi[1] == "p":
                    haars.append(Sample(f, haar[0:], PN.Positive, 0))
                else:
                    haars.append(Sample(f, haar[0:], PN.Negative, 0))
        # p/n count
        positive_count = 0
        negative_count = 0
        for i_s in range(len(haars)):
            if haars[i_s].pn == PN.Positive:
                positive_count += 1
            else:
                negative_count += 1
        if positive_count == 0 or negative_count == 0:
            raise Exception("train data set not completed")
        # init weight
        w_p = 1 / positive_count
        w_n = 1 / negative_count
        for i_s in range(len(haars)):
            if haars[i_s].pn == PN.Positive:
                haars[i_s].weight = w_p
            else:
                haars[i_s].weight = w_n
        __cache_obj(cache_path, haars, __json_encode_sample)
    return haars


def __find_weak_classifier(samples_f):
    """find the weak classifier for this sample

    :param sample_f: [SampleFeature,]
    :return:
    """
    samples_f.sort(key=lambda item: item.feature)
    min_err_curr_feature_symbol = Symbol.LtEq
    min_err_curr_feature_err = 1
    min_err_curr_feature_si = -1
    for si in range(0, len(samples_f)):
        positive_f_weight = 0
        positive_b_weight = 0
        negative_f_weight = 0
        negative_b_weight = 0
        if si + 1 < len(samples_f) and samples_f[si].feature == samples_f[si + 1].feature:
            continue
        for si_f in range(0, si + 1):
            if samples_f[si_f].pn == PN.Positive:
                positive_f_weight += samples_f[si_f].weight
            else:
                negative_f_weight += samples_f[si_f].weight
        for si_b in range(si + 1, len(samples_f)):
            if samples_f[si_b].pn == PN.Positive:
                positive_b_weight += samples_f[si_b].weight
            else:
                negative_b_weight += samples_f[si_b].weight
        err_gt = positive_f_weight + negative_b_weight
        err_lteq = negative_f_weight + positive_b_weight
        if err_lteq <= err_gt:
            c_err = err_lteq
            symbol = Symbol.LtEq
        else:
            c_err = err_gt
            symbol = Symbol.Gt
        if c_err == 0:
            print(c_err)
        if c_err <= min_err_curr_feature_err:
            min_err_curr_feature_err = c_err
            min_err_curr_feature_si = si
            min_err_curr_feature_symbol = symbol
    print("__find_weak_classifier: min_err=" + str(min_err_curr_feature_err) +
          ", min_err index=" + str(min_err_curr_feature_si))
    # (feature_sep, gt, err)
    return samples_f[min_err_curr_feature_si].feature, min_err_curr_feature_symbol, min_err_curr_feature_err


def __classify_true(sample, w_classifier):
    """return True if TP or TN

    """
    if sample.pn == PN.Positive:
        # 正例
        if w_classifier.symbol == Symbol.LtEq:
            return sample.features[w_classifier.feature_index] <= w_classifier.feature_sep
        else:
            return sample.features[w_classifier.feature_index] > w_classifier.feature_sep
    else:
        # 反例
        if w_classifier.symbol == Symbol.LtEq:
            return sample.features[w_classifier.feature_index] > w_classifier.feature_sep
        else:
            return sample.features[w_classifier.feature_index] <= w_classifier.feature_sep


def train_entry():
    # cal all dataset feature
    haars = __init_current_haars_with_weight()
    if haars is not None and len(haars) > 0:
        # features count
        feature_counts = len(haars[0].features)
        best_weak_classifier = []
        # for every time
        for t in range(200):
            # find the best weak classifier
            min_err_weak_classifier = None
            # every feature
            # for i in range(0, feature_counts, 16000):
            for i in range(feature_counts):
                # find weak classifier for each feature
                # (feature, pn, weight)
                samples_f = [SampleFeature(haars[j], i) for j in range(len(haars))]
                (feature_sep, symbol, err) = __find_weak_classifier(samples_f)
                if min_err_weak_classifier is None or err <= min_err_weak_classifier.err:
                    min_err_weak_classifier = WeakClassifier(i, feature_sep, symbol, err)
            best_weak_classifier.append(min_err_weak_classifier)
            # update alpha
            # (i, feature_sep, gt, err)
            err = min_err_weak_classifier.err
            beta = err / (1 - err)
            alpha = np.math.log(1 / beta)
            min_err_weak_classifier.alpha = alpha
            # update weight
            total_weight = 0
            for sample in haars:
                if __classify_true(sample, min_err_weak_classifier):
                    sample.weight *= beta
                total_weight += sample.weight
            for sample in haars:
                sample.weight /= total_weight
            print("train_entry: ===========Loop t=" + str(t) +
                  ", beta=" + str(beta))
            print("train_entry: ===========Loop t=" + str(t) +
                  ", total_weight=" + str(total_weight))
            print("train_entry: ===========Loop t=" + str(t) +
                  ", min_err_weak_classifier=" + str(min_err_weak_classifier))
        __cache_obj("../dataset/train/haar_classifier.model", best_weak_classifier, __json_encode_classifier)


if __name__ == '__main__':
    print("__main__: train begin")
    train_entry()
    print("__main__: train end")
