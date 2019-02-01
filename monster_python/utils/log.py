import time


class Log:
    @staticmethod
    def d(args):
        ct = time.time()
        print("%s%s D/%s" % (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ct)), str(ct-int(ct))[1:4], str(args)))
