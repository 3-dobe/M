import json
import threading
from enum import Enum
from multiprocessing import Queue, Process

import numpy as np
import os
import re

from PIL import Image

from dataset import normalize
from haar_adaboost import Haar

ROOT_DIR = "E:/workspace/chaopei/M/monster_python/"


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


def __init_current_haars_with_weight():
    cache_path = ROOT_DIR + normalize.TRAIN_DATA_INIT
    haars = __get_cache_obj(cache_path, __json_decode_sample)
    if haars is None:
        train_data_dir = ROOT_DIR + normalize.TRAIN_NORM_DIR
        # tuple list
        haars = []
        for f in os.listdir(train_data_dir):
            if f.startswith("norm_") and f.endswith(".png"):
                haar = Haar(f, np.array(Image.open(train_data_dir + f, "r")))
                haar.cal()
                hi = re.split("[_.]", f)
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


def __process_entry_weak_classify(haars, f_start, feature_counts, feature_counts_process, queue):
    min_err_weak_classifier = None
    for i in range(f_start, f_start + feature_counts_process):
        if i >= feature_counts:
            break
        # find weak classifier for each feature
        # (feature, pn, weight)
        samples_f = [SampleFeature(haars[j], i) for j in range(len(haars))]
        (feature_sep, symbol, err) = __find_weak_classifier(samples_f)
        if min_err_weak_classifier is None or err <= min_err_weak_classifier.err:
            min_err_weak_classifier = WeakClassifier(i, feature_sep, symbol, err)
        print("__thread_entry_weak_classify: thread=" + str(threading.current_thread()) +
              ", i=" + str(i) +
              ", f_start=" + str(f_start) +
              " ~ f_start+feature_counts_thread=" + str(f_start + feature_counts_process))
    print("__thread_entry_weak_classify: thread" + str(threading.current_thread()) +
          ", " + str(min_err_weak_classifier))
    queue.put(min_err_weak_classifier)


def __train_entry(haars, loop_t=1, thread_count=1):
    if haars is not None and len(haars) > 0:
        # features count
        feature_counts = len(haars[0].features)
        # feature_counts = int(len(haars[0].features) / 100)
        best_weak_classifier = []
        # for every time
        for t in range(loop_t):
            print("train_entry: BEGIN===========Loop t=" + str(t))
            # find the best weak classifier
            min_err_weak_classifier = None
            # every feature
            feature_counts_process = np.math.ceil(feature_counts / thread_count)
            processes = []
            # for i in range(0, feature_counts, 16000):
            result_q = Queue()
            for i in range(0, feature_counts, feature_counts_process):
                p = Process(target=__process_entry_weak_classify,
                            args=(haars, i, feature_counts, feature_counts_process, result_q))
                processes.append(p)
                p.start()
            for i in range(len(processes)):
                processes[i].join()
                print("train_entry: thread " + str(i) + " join")
                res = result_q.get_nowait()
                if min_err_weak_classifier is None or res[0].err <= min_err_weak_classifier.err:
                    min_err_weak_classifier = res[0]
            print("train_entry: Loop t=" + str(t) + ", best_weak_classifier found, " + str(min_err_weak_classifier))
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
            print("train_entry: END=============Loop t=" + str(t) +
                  ", beta=" + str(beta) +
                  ", total_weight=" + str(total_weight) +
                  ", min_err_weak_classifier=" + str(min_err_weak_classifier))
        __cache_obj(ROOT_DIR + normalize.TRAIN_MODEL,
                    best_weak_classifier,
                    __json_encode_classifier)


if __name__ == '__main__':
    # cal all dataset feature
    print("__main__: init data begin")
    haars = __init_current_haars_with_weight()
    print("__main__: init data end")
    # train
    print("__main__: train begin")
    __train_entry(haars, 200, 8)
    print("__main__: train end")
