from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier


class EmotionClassifierTest:

    def __init__(self, winner_classifier: MLPClassifier):
        self.__classifier = winner_classifier
        self.__test_y = None
        self.__test_x = None
        self.__validation_x = None
        self.__validation_y = None

    def set_validation_test_set(self, training_test_validation_dict: dict):
        test_set = training_test_validation_dict['testing']['set']
        validation_set = training_test_validation_dict['validation']['set']

        self.__test_x = []
        self.__test_y = []
        for sample in test_set:
            self.__test_x.append(sample[:-1])
            self.__test_y.append(sample[-1])

        self.__validation_x = []
        self.__validation_y = []
        for sample in validation_set:
            self.__validation_x.append(sample[:-1])
            self.__validation_y.append(sample[-1])

    def test_winner(self):
        prediction_test = self.__classifier.predict(self.__test_x)
        accuracy_test = accuracy_score(self.__test_y, prediction_test)
        prediction_validation = self.__classifier.predict(self.__validation_x)
        accuracy_validation = accuracy_score(self.__validation_y, prediction_validation)
        return [accuracy_test, accuracy_validation]
