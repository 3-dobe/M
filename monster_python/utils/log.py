import time


class Log:
    @staticmethod
    def d(args):
        ct = time.time()
        print("\033[0;30;m%s%s D/%s\033[0m" % (
            time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ct)), str(ct - int(ct))[1:4], str(args)))

    @staticmethod
    def e(args):
        ct = time.time()
        print("\033[0;31;m%s%s E/%s\033[0m" % (
            time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ct)), str(ct - int(ct))[1:4], str(args)))

    @staticmethod
    def w(args):
        ct = time.time()
        print("\033[0;34;m%s%s W/%s\033[0m" % (
            time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ct)), str(ct - int(ct))[1:4], str(args)))
