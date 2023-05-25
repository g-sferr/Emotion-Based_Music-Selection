import matplotlib.pyplot as plt


from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier


class EmotionClassifierTraining:

    def __init__(self, max_neurons: int, min_neurons: int, step_neurons: int, min_layers: int, max_layers: int,
                 step_layers: int, start_number_epochs: int):
        self.__max_neurons = max_neurons
        self.__min_neurons = min_neurons
        self.__step_neurons = step_neurons
        self.__min_layers = min_layers
        self.__max_layers = max_layers
        self.__step_layers = step_layers
        self.__avg_neurons = 0
        self.__avg_layers = 0
        self.__start_number_epochs = start_number_epochs
        self.__classifier = None
        self.__training_x = None
        self.__validation_x = None
        self.__training_y = None
        self.__validation_y = None

    def set_average_hyper_parameters(self):
        self.__avg_neurons = round((self.__max_neurons + self.__min_neurons) / 2)
        self.__avg_layers = round((self.__max_layers + self.__min_layers) / 2)
        architecture = list()
        for _ in range(self.__avg_layers):
            architecture.append(self.__avg_neurons)
        self.__classifier = MLPClassifier(hidden_layer_sizes=(tuple(architecture)), max_iter=self.__start_number_epochs)

    def training(self):
        print(self.__training_x)
        print(self.__training_y)
        self.__classifier.fit(self.__training_x, self.__training_y)

    def set_validation_training_set(self, training_test_validation_dict: dict):
        print(training_test_validation_dict['training'])
        training_set = training_test_validation_dict['training']['set']
        validation_set = training_test_validation_dict['validation']['set']

        self.__training_x = []
        self.__training_y = []
        for sample in training_set:
            self.__training_x.append(sample[:-1])
            self.__training_y.append(sample[-1])

        self.__validation_x = []
        self.__validation_y = []
        for sample in validation_set:
            self.__validation_x.append(sample[:-1])
            self.__validation_y.append(sample[-1])

    def generate_plot(self):
        plt.plot(self.__classifier.loss_curve_)
        plt.title("MLPClassifier Training Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.show()
        plt.savefig("./Plot/training_loss.png")

    def grid_search(self):
        self.__classifier = MLPClassifier()
        hidden_layers = []
        for i in range(self.__min_layers, self.__max_layers + 1, self.__step_layers):
            for j in range(self.__min_neurons, self.__max_neurons, self.__step_neurons):
                tmp_list = []
                for _ in range(i):
                    tmp_list.append(j)
                hidden_layers.append(tuple(tmp_list))
        param_grid = {'hidden_layer_sizes': hidden_layers}
        # n_samples = len(self.__training_x)
        grid_search = GridSearchCV(estimator=self.__classifier, param_grid=param_grid, cv=2)
        grid_search.fit(self.__training_x, self.__training_y)

        best_classifiers = [grid_search.best_estimator_]
        best_scores = [grid_search.best_score_]
        for mean_test_score, params in sorted(
                zip(grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['params']), key=lambda x: x[0],
                reverse=True):
            if len(best_classifiers) >= 5:
                break
            clf = MLPClassifier(**params)
            clf.fit(self.__training_x, self.__training_y)
            best_classifiers.append(clf)
            best_scores.append(mean_test_score)

        best = list(zip(best_classifiers, best_scores))

        return best

    def test_winner_training(self, winner: MLPClassifier):
        prediction_training = winner.predict(self.__training_x)
        accuracy_training = accuracy_score(self.__training_y, prediction_training)
        prediction_validation = winner.predict(self.__validation_x)
        accuracy_validation = accuracy_score(self.__validation_y, prediction_validation)
        return [accuracy_training, accuracy_validation]
