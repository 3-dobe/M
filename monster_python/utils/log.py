import sys


class Log:
    @staticmethod
    def d(*objects, sep=' ', end='\n', file=sys.stdout):
        print(objects, sep, end, file)
