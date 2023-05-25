import json

from development_system import DevelopmentSystem
from rest_server import RestServer


def main():
    development_system = DevelopmentSystem(development_system_parameters)
    if development_system_parameters['execution_mode'] == "first_training":
        rest_server = RestServer("rest_server",
                                 develop_sys=development_system)
        rest_server.run(host=development_system_parameters["development_system"]["ip"],
                        port=development_system_parameters["development_system"]["port"],
                        debug=True)
    elif development_system_parameters['execution_mode'] == "training":
        development_system.training_mode()
    elif development_system_parameters['execution_mode'] == "grid_search":
        development_system.grid_search()
    elif development_system_parameters['execution_mode'] == "final_test":
        development_system.final_test()
    else:
        development_system.deploy_and_delete()


if __name__ == "__main__":
    with open("./Configuration/development_system_configuration.json", "r") as configuration_file:
        development_system_parameters = json.load(configuration_file)
    main()
