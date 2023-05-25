import json
import pickle
import random

from development_database import DevelopmentDatabase
from emotion_classifier_test import EmotionClassifierTest
from emotion_classifier_training import EmotionClassifierTraining
from json_validator import JsonValidator
from rest_server import RestServer


class DevelopmentSystem:
    def __init__(self, development_system_parameters: dict):
        self.__development_system_parameters = development_system_parameters
        self.__json_validator = JsonValidator()
        if self.__development_system_parameters['execution_mode'] == "final_test":
            self.__emotion_classifier = None
        else:
            self.__emotion_classifier = EmotionClassifierTraining(
                self.__development_system_parameters['max_value_neurons'],
                self.__development_system_parameters['min_value_neurons'],
                self.__development_system_parameters['step_neurons'],
                self.__development_system_parameters['min_value_layers'],
                self.__development_system_parameters['max_value_layers'],
                self.__development_system_parameters['step_layers'],
                self.__development_system_parameters['start_number'
                                                     '_epochs'])
        self.__development_database = DevelopmentDatabase('./Database/training_test_db')
        if self.__development_system_parameters['testing']:
            self.__number_classifier_deployed = 0

    def first_training_mode(self, received_json: dict):
        if self.__development_system_parameters['testing']:
            self.__emotion_classifier = EmotionClassifierTraining(
                self.__development_system_parameters['max_value_neurons'],
                self.__development_system_parameters['min_value_neurons'],
                self.__development_system_parameters['step_neurons'],
                self.__development_system_parameters['min_value_layers'],
                self.__development_system_parameters['max_value_layers'],
                self.__development_system_parameters['step_layers'],
                self.__development_system_parameters['start_number'
                                                     '_epochs'])
        if not self.__json_validator.validate(received_json):
            raise RuntimeError
        self.__development_database.store_json(received_json)
        self.__emotion_classifier.set_average_hyper_parameters()
        print(received_json)
        self.__emotion_classifier.set_validation_training_set(received_json)
        self.__emotion_classifier.training()
        if self.__development_system_parameters['testing']:
            self.training_mode()
        else:
            self.__emotion_classifier.generate_plot()

    def training_mode(self):
        test_training_validation_set = self.__development_database.get_json()
        self.__emotion_classifier.set_average_hyper_parameters()
        self.__emotion_classifier.set_validation_training_set(test_training_validation_set)
        self.__emotion_classifier.training()
        if self.__development_system_parameters['testing']:
            self.grid_search()
        else:
            self.__emotion_classifier.generate_plot()

    def grid_search(self):
        test_training_validation_set = self.__development_database.get_json()
        self.__emotion_classifier.set_validation_training_set(test_training_validation_set)
        top_5_classifier = self.__emotion_classifier.grid_search()
        old_top_5_classifier = self.__development_database.get_list()
        if old_top_5_classifier is None:
            self.__development_database.store_list(top_5_classifier)
        else:
            new_top_5_classifier = top_5_classifier + old_top_5_classifier
            sorted_list = sorted(new_top_5_classifier, key=lambda x: x[1])
            new_top_5_classifier = sorted_list[:5]
            top_classifier = sorted_list[0][0]
            self.__save_accuracy(top_classifier)
            self.__development_database.store_list(new_top_5_classifier)
        if self.__development_system_parameters['testing']:
            random_number = random.random()
            if random_number > 0.3:
                self.__number_classifier_deployed += 1
                self.final_test()
            else:
                self.grid_search()

    def __save_accuracy(self, top_classifier):
        [accuracy_training, accuracy_validation] = self.__emotion_classifier.test_winner_training(top_classifier)
        accuracy_json = {'accuracy_training': accuracy_training, 'accuracy_validation': accuracy_validation}
        with open("./Accuracy/accuracy_validation_training.json", "w") as accuracy_json_file:
            json.dump(accuracy_json, accuracy_json_file)

    def final_test(self):
        test_training_validation_set = self.__development_database.get_json()
        winner_classifier = self.__development_database.get_list()[0][0]
        self.__emotion_classifier = EmotionClassifierTest(winner_classifier)
        self.__emotion_classifier.set_validation_test_set(test_training_validation_set)
        [accuracy_test, accuracy_validation] = self.__emotion_classifier.test_winner()
        accuracy_json = {'accuracy_test': accuracy_test, 'accuracy_validation': accuracy_validation}
        with open("./Accuracy/accuracy_validation_test.json", "w") as accuracy_json_file:
            json.dump(accuracy_json, accuracy_json_file)
        if self.__development_system_parameters['testing']:
            self.deploy_and_delete()

    def deploy_and_delete(self):
        winner_classifier = self.__development_database.get_list()[0][0]
        with open('./Classifier/classifier.sav', 'wb') as classifier:
            pickle.dump(winner_classifier, classifier)
        RestServer.send_file('./Classifier/classifier.sav',
                             self.__development_system_parameters['execution_system']['ip'],
                             self.__development_system_parameters['execution_system']['port'], '/send_classifier')
        if not self.__development_system_parameters['testing']:
            self.__development_database.del_json()
            self.__development_database.del_list()
