import numpy as np

from utils import Log


class Haar:
    def __init__(self, name, img, ratio=1):
        """ init with a data array of img, final feature window width is width(img)/ratio

        :param img: L img array
        """
        if img is None:
            raise Exception("Haar input img must be an NxN array")
        self.__name = name
        self.__img = img
        self.__ig = Haar.obtain_integral_graph(img)
        self.__ratio = ratio
        self.__features = None

    def __extract(self):
        """extract the haar-like feature

        """
        haar_features_1 = self.__cal_haar_1_features()
        haar_features_2 = self.__cal_haar_2_features()
        haar_features_3 = self.__cal_haar_3_features()
        haar_features_4 = self.__cal_haar_4_features()
        haar_features_5 = self.__cal_haar_5_features()
        haar_features = haar_features_1 + haar_features_2 + haar_features_3 + haar_features_4 + haar_features_5
        Log.d("__extract:" + self.__name + " haar_features count " + str(len(haar_features)))
        # Log.d(haar_features)
        return haar_features

    @staticmethod
    def obtain_integral_graph(img):
        """calculate integral img

        :param img:
        :return:
        """
        integ_graph = np.zeros((img.shape[0], img.shape[1]), dtype=np.int32)
        for x in range(img.shape[0]):
            sum_clo = 0
            for y in range(img.shape[1]):
                sum_clo = sum_clo + img[x][y]
                integ_graph[x][y] = integ_graph[x - 1][y] + sum_clo
        return integ_graph

    def __cal_by_integral_graph(self, lt, rt, lb, rb):
        """指定左上/右上/左下/右下四个点坐标, 计算该区块像素和, 坐标(x,y)对应数组是[y][x], y行x列

        :param ig:
        :param lt:
        :param rt:
        :param lb:
        :param rb:
        :return:
        """
        if lt[0] >= 0 and lt[1] >= 0:
            lt_img = self.__ig[self.__ratio * (lt[1] + 1) - 1][self.__ratio * (lt[0] + 1) - 1]
        else:
            lt_img = 0
        if rt[0] >= 0 and rt[1] >= 0:
            rt_img = self.__ig[self.__ratio * (rt[1] + 1) - 1][self.__ratio * (rt[0] + 1) - 1]
        else:
            rt_img = 0
        if lb[0] >= 0 and lb[1] >= 0:
            lb_img = self.__ig[self.__ratio * (lb[1] + 1) - 1][self.__ratio * (lb[0] + 1) - 1]
        else:
            lb_img = 0
        if rb[0] >= 0 and rb[1] >= 0:
            rb_img = self.__ig[self.__ratio * (rb[1] + 1) - 1][self.__ratio * (rb[0] + 1) - 1]
        else:
            rb_img = 0
        return int((lt_img - rt_img - lb_img + rb_img) / self.__ratio / self.__ratio)

    def __cal_haar_1_features(self):
        """ Types of Haar-like rectangle features
        -----
        | + |
        | - |
        -----
        calculate haar features

        :return:
        """
        (width, height) = (int(self.__img.shape[0] / self.__ratio), int(self.__img.shape[1] / self.__ratio))
        width_limit = width
        height_limit = height / 2
        # features_graph 四元组 (x, y, w, h), 取值范围 x {0 ~ width-1}, w {1 ~ width}
        features_graph = []
        for w in range(1, int(width_limit + 1)):
            for h in range(1, int(height_limit + 1)):
                w_move_limit = width - w
                h_move_limit = height - 2 * h
                for x in range(0, w_move_limit + 1):
                    for y in range(0, h_move_limit + 1):
                        features_graph.append([x, y, w, h])
        haar_features = []
        for num in range(len(features_graph)):
            (x, y, w, h) = (
                features_graph[num][0], features_graph[num][1], features_graph[num][2], features_graph[num][3])
            # 计算上面的矩形区局的像素和
            haar1 = self.__cal_by_integral_graph(
                (x - 1, y - 1),
                (x + w - 1, y - 1),
                (x - 1, y + h - 1),
                (x + w - 1, y + h - 1))
            # 计算下面的矩形区域的像素和
            haar2 = self.__cal_by_integral_graph(
                (x - 1, y + h - 1),
                (x + w - 1, y + h - 1),
                (x - 1, y + 2 * h - 1),
                (x + w - 1, y + 2 * h - 1))
            # 上面的像素和减去下面的像素和
            haar_features.append(haar1 - haar2)
        return haar_features

    def __cal_haar_2_features(self):
        """ Types of Haar-like rectangle features
        ---------
        | + | - |
        ---------
        calculate haar features

        :return:
        """
        (width, height) = (int(self.__img.shape[0] / self.__ratio), int(self.__img.shape[1] / self.__ratio))
        width_limit = width / 2
        height_limit = height
        features_graph = []
        for w in range(1, int(width_limit + 1)):
            for h in range(1, int(height_limit + 1)):
                w_move_limit = width - 2 * w
                h_move_limit = height - h
                for x in range(0, w_move_limit + 1):
                    for y in range(0, h_move_limit + 1):
                        features_graph.append([x, y, w, h])
        haar_features = []
        for num in range(len(features_graph)):
            (x, y, w, h) = (
                features_graph[num][0], features_graph[num][1], features_graph[num][2], features_graph[num][3])
            # 计算左面的矩形区局的像素和
            haar1 = self.__cal_by_integral_graph(
                (x - 1, y - 1),
                (x + w - 1, y - 1),
                (x - 1, y + h - 1),
                (x + w - 1, y + h - 1))
            # 计算右面的矩形区域的像素和
            haar2 = self.__cal_by_integral_graph(
                (x + w - 1, y - 1),
                (x + 2 * w - 1, y - 1),
                (x + w - 1, y + h - 1),
                (x + 2 * w - 1, y + h - 1))
            # 左面的像素和减去右面的像素和
            haar_features.append(haar1 - haar2)
        return haar_features

    def __cal_haar_3_features(self):
        """ Types of Haar-like rectangle features
        -------------
        | + | - | + |
        -------------
        calculate haar features

        :return:
        """
        (width, height) = (int(self.__img.shape[0] / self.__ratio), int(self.__img.shape[1] / self.__ratio))
        width_limit = width / 3
        height_limit = height
        features_graph = []
        for w in range(1, int(width_limit + 1)):
            for h in range(1, int(height_limit + 1)):
                w_move_limit = width - 3 * w
                h_move_limit = height - h
                for x in range(0, w_move_limit + 1):
                    for y in range(0, h_move_limit + 1):
                        features_graph.append([x, y, w, h])
        haar_features = []
        for num in range(len(features_graph)):
            (x, y, w, h) = (
                features_graph[num][0], features_graph[num][1], features_graph[num][2], features_graph[num][3])
            # 计算左面的矩形区局的像素和
            haar1 = self.__cal_by_integral_graph(
                (x - 1, y - 1),
                (x + w - 1, y - 1),
                (x - 1, y + h - 1),
                (x + w - 1, y + h - 1))
            # 计算中间的矩形区域的像素和
            haar2 = self.__cal_by_integral_graph(
                (x + w - 1, y - 1),
                (x + 2 * w - 1, y - 1),
                (x + w - 1, y + h - 1),
                (x + 2 * w - 1, y + h - 1))
            # 计算右面的矩形区域的像素和
            haar3 = self.__cal_by_integral_graph(
                (x + 2 * w - 1, y - 1),
                (x + 3 * w - 1, y - 1),
                (x + 2 * w - 1, y + h - 1),
                (x + 3 * w - 1, y + h - 1))
            haar_features.append(haar3 + haar1 - 2 * haar2)
        return haar_features

    def __cal_haar_4_features(self):
        """ Types of Haar-like rectangle features
        -----
        | + |
        | - |
        | + |
        -----
        calculate haar features

        :return:
        """
        (width, height) = (int(self.__img.shape[0] / self.__ratio), int(self.__img.shape[1] / self.__ratio))
        width_limit = width
        height_limit = height / 3
        features_graph = []
        for w in range(1, int(width_limit + 1)):
            for h in range(1, int(height_limit + 1)):
                w_move_limit = width - w
                h_move_limit = height - 3 * h
                for x in range(0, w_move_limit + 1):
                    for y in range(0, h_move_limit + 1):
                        features_graph.append([x, y, w, h])
        haar_features = []
        for num in range(len(features_graph)):
            (x, y, w, h) = (
                features_graph[num][0], features_graph[num][1], features_graph[num][2], features_graph[num][3])
            # 计算上面的矩形区局的像素和
            haar1 = self.__cal_by_integral_graph(
                (x - 1, y - 1),
                (x + w - 1, y - 1),
                (x - 1, y + h - 1),
                (x + w - 1, y + h - 1))
            # 计算中间的矩形区域的像素和
            haar2 = self.__cal_by_integral_graph(
                (x - 1, y + h - 1),
                (x + w - 1, y + h - 1),
                (x - 1, y + 2 * h - 1),
                (x + w - 1, y + 2 * h - 1))
            # 计算下面的矩形区域的像素和
            haar3 = self.__cal_by_integral_graph(
                (x - 1, y + 2 * h - 1),
                (x + w - 1, y + 2 * h - 1),
                (x - 1, y + 3 * h - 1),
                (x + w - 1, y + 3 * h - 1))
            haar_features.append(haar3 + haar1 - 2 * haar2)
        return haar_features

    def __cal_haar_5_features(self):
        """ Types of Haar-like rectangle features
        -----
        |+|-|
        |-|+|
        -----
        calculate haar features

        :return:
        """
        (width, height) = (int(self.__img.shape[0] / self.__ratio), int(self.__img.shape[1] / self.__ratio))
        width_limit = width / 2
        height_limit = height / 2
        features_graph = []
        for w in range(1, int(width_limit + 1)):
            for h in range(1, int(height_limit + 1)):
                w_move_limit = width - 2 * w
                h_move_limit = height - 2 * h
                for x in range(0, w_move_limit + 1):
                    for y in range(0, h_move_limit + 1):
                        features_graph.append([x, y, w, h])
        haar_features = []
        for num in range(len(features_graph)):
            (x, y, w, h) = (
                features_graph[num][0], features_graph[num][1], features_graph[num][2], features_graph[num][3])
            # 计算左上的矩形区局的像素和
            haar1 = self.__cal_by_integral_graph(
                (x - 1, y - 1),
                (x + w - 1, y - 1),
                (x - 1, y + h - 1),
                (x + w - 1, y + h - 1))
            # 计算右上的矩形区域的像素和
            haar2 = self.__cal_by_integral_graph(
                (x + w - 1, y - 1),
                (x + 2 * w - 1, y - 1),
                (x + w - 1, y + h - 1),
                (x + 2 * w - 1, y + h - 1))
            # 计算左下的矩形区域的像素和
            haar3 = self.__cal_by_integral_graph(
                (x - 1, y + h - 1),
                (x + w - 1, y + h - 1),
                (x - 1, y + 2 * h - 1),
                (x + w - 1, y + 2 * h - 1))
            # 计算右下的矩形区域的像素和
            haar4 = self.__cal_by_integral_graph(
                (x + w - 1, y + h - 1),
                (x + 2 * w - 1, y + h - 1),
                (x + w - 1, y + 2 * h - 1),
                (x + 2 * w - 1, y + 2 * h - 1))
            haar_features.append(haar4 + haar1 - haar3 - haar2)
        return haar_features

    def cal(self):
        if self.__features is None:
            self.__features = self.__extract()

    def __getitem__(self, key):
        if self.__features is None:
            self.__features = self.__extract()
        return self.__features[key]

    def __setitem__(self):
        raise Exception("not support setter")

    def __delitem__(self):
        raise Exception("not support del")

    def __len__(self):
        if self.__features is None:
            self.__features = self.__extract()
        return len(self.__features)
