from typing import List

import cv2
import numpy as np

x = cv2.face.EigenFaceRecognizer_create(1)
from person import Person

class PCAEigenFace:

    def __calcMean(train):
        sample = train[0].data
        mean = np.zeros((sample.rows, sample.cols), dtype=np.uint8)


    def __calcDiff(train):
        pass

    def __calcCovariance():
        pass

    def __calcEigen():
        pass

    def __calcEigenFaces():
        pass

    def __calcProjections(train):
        pass

    def __init__(self, numComponents: int):
       self.numComponents = numComponents
       self.mean = None
       self.diffs = None
       self.covariance = None
       self.eigenvalues = None
       self.eigenvectors = None
       self.eigenfaces = None
       self.projections = None
       self.labels = []


    def train(self, train: List[Person]):
        self.__calcMean(train)
        self.__calcDiff(train)
        self.__calcCovariance()
        self.__calcEigen()
        self.__calcEigenFaces()
        self.__calcProjections(train)



