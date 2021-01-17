import os
import random

import numpy as np
from cv2 import cv2
from sklearn.metrics import accuracy_score

from person import Person


def load_people_from_folder(folder):
    people = []
    for filename in os.listdir(folder):
        person = Person.to_person(folder, filename)
        if person:
            people.append(person)
    return people


def load_dataset(directory):
    train_data = []
    test_data = []
    people = load_people_from_folder(directory)
    people.sort(key=lambda x: int(x.id))

    for i in range(0, len(people), 10):
        person = people[i:i + 10]
        indices_treino = random.sample(range(10), k=7)

        for j in range(10):
            if j in indices_treino:
                train_data.append(person[j])
            else:
                test_data.append(person[j])

    return train_data, test_data


def model(number_of_components, train_data, test_data):
    model = cv2.face.EigenFaceRecognizer_create(number_of_components)

    faces = []
    labels = []

    for item in train_data:
        faces.append(item.data)
        labels.append(item.label)

    model.train(faces, np.array(labels))

    faces = []
    labels = []

    for item in test_data:
        faces.append(item.data)
        labels.append(item.label)

    prediction = list()

    for i, image in enumerate(faces):
        label, confidence = model.predict(image)
        prediction.append(label)

    accuracy = accuracy_score(prediction, labels)

    return number_of_components, round(accuracy * 100, 2)