import numpy as np

from artm import EPS


class LogFunction(object):
    @staticmethod
    def calc(x):
        return np.log(x + EPS)

    @staticmethod
    def calc_der(x):
        return 1. / (x + EPS)


class IdFunction(object):
    @staticmethod
    def calc(x):
        return x + EPS

    @staticmethod
    def calc_der(x):
        return np.ones_like(x)


class SquareFunction(object):
    @staticmethod
    def calc(x):
        return (x + EPS) ** 2

    @staticmethod
    def calc_der(x):
        return 2. * (x + EPS) ** 2


class CubeLogFunction(object):
    @staticmethod
    def calc(x):
        return np.log(x + EPS) ** 3

    @staticmethod
    def calc_der(x):
        return 3. * np.log(x + EPS) ** 2 / (x + EPS)


class SquareLogFunction(object):
    @staticmethod
    def calc(x):
        return np.log(x + EPS) * np.abs(np.log(x + EPS))

    @staticmethod
    def calc_der(x):
        return 2. * np.abs(np.log(x + EPS)) / (x + EPS)


class FiveLogFunction(object):
    @staticmethod
    def calc(x):
        return np.log(x + EPS) ** 5

    @staticmethod
    def calc_der(x):
        return 5. * np.log(x + EPS) ** 4 / (x + EPS)


class CubeRootLogFunction(object):
    @staticmethod
    def calc(x):
        return np.cbrt(np.log(x + EPS))

    @staticmethod
    def calc_der(x):
        return 1. / 3 / (np.cbrt(np.log(x + EPS)) ** 2) / (x + EPS)


class SquareRootLogFunction(object):
    @staticmethod
    def calc(x):
        return np.sqrt(- np.log(x + EPS))

    @staticmethod
    def calc_der(x):
        return 0.5 / np.sqrt(- np.log(x + EPS)) / (x + EPS)


class ExpFunction(object):
    @staticmethod
    def calc(x):
        return np.exp(x)

    @staticmethod
    def calc_der(x):
        return np.exp(x)


class EntropyFunction(object):
    @staticmethod
    def calc(x):
        return (np.log(x + EPS)) * (x + EPS)

    @staticmethod
    def calc_der(x):
        return np.log(x + EPS)
