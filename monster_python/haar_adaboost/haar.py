import numpy as np


class Haar:
    def __init__(self, img):
        """ init with a data array of img

        :param img: img array
        """
        if img is None:
            raise Exception("Haar input img must be an NxN array")
        self.__img = img
        self.__features = None

    def __extract(self):
        """extract the haar-like feature

        """
        integral_graph = self.__get_integral_graph(self.__img)
        hear_features_1 = self.__cal_haar_1_features(integral_graph, self.__img.shape[0], self.__img.shape[1])
        hear_features_2 = self.__cal_haar_2_features(integral_graph, self.__img.shape[0], self.__img.shape[1])
        hear_features_3 = self.__cal_haar_3_features(integral_graph, self.__img.shape[0], self.__img.shape[1])
        hear_features_4 = self.__cal_haar_4_features(integral_graph, self.__img.shape[0], self.__img.shape[1])
        hear_features_5 = self.__cal_haar_5_features(integral_graph, self.__img.shape[0], self.__img.shape[1])
        print("Haar.__extract: hear_features_1 count " + str(len(hear_features_1)))
        print("Haar.__extract: hear_features_2 count " + str(len(hear_features_2)))
        print("Haar.__extract: hear_features_3 count " + str(len(hear_features_3)))
        print("Haar.__extract: hear_features_4 count " + str(len(hear_features_4)))
        print("Haar.__extract: hear_features_5 count " + str(len(hear_features_5)))
        return hear_features_1 + hear_features_2 + hear_features_3 + hear_features_4 + hear_features_5

    @staticmethod
    def __get_integral_graph(img):
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

    @staticmethod
    def __cal_by_integral_graph(ig, lt, rt, lb, rb):
        """指定左上/右上/左下/右下四个点, 计算该区块像素和

        :param ig:
        :param lt:
        :param rt:
        :param lb:
        :param rb:
        :return:
        """
        if lt[0] >= 0 and lt[1] >= 0:
            lt_img = ig[lt[0]][lt[1]]
        else:
            lt_img = 0
        if rt[0] >= 0 and rt[1] >= 0:
            rt_img = ig[rt[0]][rt[1]]
        else:
            rt_img = 0
        if lb[0] >= 0 and lb[1] >= 0:
            lb_img = ig[lb[0]][lb[1]]
        else:
            lb_img = 0
        if rb[0] >= 0 and rb[1] >= 0:
            rb_img = ig[rb[0]][rb[1]]
        else:
            rb_img = 0
        return lt_img - rt_img - lb_img + rb_img

    @staticmethod
    def __cal_haar_1_features(ig, width, height):
        """ Types of Haar-like rectangle features
        -----
        | + |
        | - |
        -----
        calculate hear features

        :param width:
        :param height:
        :return:
        """
        width_limit = width
        height_limit = height / 2
        # integral_graph 四元组 (x, y, w, h), 取值范围 x {0 ~ width-1}, w {1 ~ width}
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
            haar1 = Haar.__cal_by_integral_graph(
                ig,
                (x - 1, y - 1),
                (x + w - 1, y - 1),
                (x - 1, y + h - 1),
                (x + w - 1, y + h - 1))
            # 计算下面的矩形区域的像素和
            haar2 = Haar.__cal_by_integral_graph(
                ig,
                (x - 1, y + h - 1),
                (x + w - 1, y + h - 1),
                (x - 1, y + 2 * h - 1),
                (x + w - 1, y + 2 * h - 1))
            # 上面的像素和减去下面的像素和
            haar_features.append(haar1 - haar2)
        return haar_features

    @staticmethod
    def __cal_haar_2_features(ig, width, height):
        """ Types of Haar-like rectangle features
        ---------
        | + | - |
        ---------
        calculate hear features

        :param width:
        :param height:
        :return:
        """
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
            haar1 = Haar.__cal_by_integral_graph(
                ig,
                (x - 1, y - 1),
                (x + w - 1, y - 1),
                (x - 1, y + h - 1),
                (x + w - 1, y + h - 1))
            # 计算右面的矩形区域的像素和
            haar2 = Haar.__cal_by_integral_graph(
                ig,
                (x + w - 1, y - 1),
                (x + 2 * w - 1, y - 1),
                (x + w - 1, y + h - 1),
                (x + 2 * w - 1, y + h - 1))
            # 左面的像素和减去右面的像素和
            haar_features.append(haar1 - haar2)
        return haar_features

    @staticmethod
    def __cal_haar_3_features(ig, width, height):
        """ Types of Haar-like rectangle features
        -------------
        | + | - | + |
        -------------
        calculate hear features

        :param width:
        :param height:
        :return:
        """
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
            haar1 = Haar.__cal_by_integral_graph(
                ig,
                (x - 1, y - 1),
                (x + w - 1, y - 1),
                (x - 1, y + h - 1),
                (x + w - 1, y + h - 1))
            # 计算中间的矩形区域的像素和
            haar2 = Haar.__cal_by_integral_graph(
                ig,
                (x + w - 1, y - 1),
                (x + 2 * w - 1, y - 1),
                (x + w - 1, y + h - 1),
                (x + 2 * w - 1, y + h - 1))
            # 计算右面的矩形区域的像素和
            haar3 = Haar.__cal_by_integral_graph(
                ig,
                (x + 2 * w - 1, y - 1),
                (x + 3 * w - 1, y - 1),
                (x + 2 * w - 1, y + h - 1),
                (x + 3 * w - 1, y + h - 1))
            haar_features.append(haar3 + haar1 - 2 * haar2)
        return haar_features

    @staticmethod
    def __cal_haar_4_features(ig, width, height):
        """ Types of Haar-like rectangle features
        -----
        | + |
        | - |
        | + |
        -----
        calculate hear features

        :param width:
        :param height:
        :return:
        """
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
            haar1 = Haar.__cal_by_integral_graph(
                ig,
                (x - 1, y - 1),
                (x + w - 1, y - 1),
                (x - 1, y + h - 1),
                (x + w - 1, y + h - 1))
            # 计算中间的矩形区域的像素和
            haar2 = Haar.__cal_by_integral_graph(
                ig,
                (x - 1, y + h - 1),
                (x + w - 1, y + h - 1),
                (x - 1, y + 2 * h - 1),
                (x + w - 1, y + 2 * h - 1))
            # 计算下面的矩形区域的像素和
            haar3 = Haar.__cal_by_integral_graph(
                ig,
                (x - 1, y + 2 * h - 1),
                (x + w - 1, y + 2 * h - 1),
                (x - 1, y + 3 * h - 1),
                (x + w - 1, y + 3 * h - 1))
            haar_features.append(haar3 + haar1 - 2 * haar2)
        return haar_features

    @staticmethod
    def __cal_haar_5_features(ig, width, height):
        """ Types of Haar-like rectangle features
        -----
        |+|-|
        |-|+|
        -----
        calculate hear features

        :param width:
        :param height:
        :return:
        """
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
            haar1 = Haar.__cal_by_integral_graph(
                ig,
                (x - 1, y - 1),
                (x + w - 1, y - 1),
                (x - 1, y + h - 1),
                (x + w - 1, y + h - 1))
            # 计算右上的矩形区域的像素和
            haar2 = Haar.__cal_by_integral_graph(
                ig,
                (x + w - 1, y - 1),
                (x + 2 * w - 1, y - 1),
                (x + w - 1, y + h - 1),
                (x + 2 * w - 1, y + h - 1))
            # 计算左下的矩形区域的像素和
            haar3 = Haar.__cal_by_integral_graph(
                ig,
                (x - 1, y + h - 1),
                (x + w - 1, y + h - 1),
                (x - 1, y + 2 * h - 1),
                (x + w - 1, y + 2 * h - 1))
            # 计算右下的矩形区域的像素和
            haar4 = Haar.__cal_by_integral_graph(
                ig,
                (x + w - 1, y + h - 1),
                (x + 2 * w - 1, y + h - 1),
                (x + w - 1, y + 2 * h - 1),
                (x + 2 * w - 1, y + 2 * h - 1))
            haar_features.append(haar4 + haar1 - haar3 - haar2)
        return haar_features

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
