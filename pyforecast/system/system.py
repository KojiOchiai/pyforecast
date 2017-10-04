# coding: utf-8

from abc import ABCMeta, abstractmethod


class AbstractSystem(metaclass=ABCMeta):
    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def observe(self):
        pass
