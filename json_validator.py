import json

from jsonschema import Draft7Validator, exceptions


class JsonValidator:

    def __init__(self):
        with open("./Schema/training_test_validation_schema.json") as training_test_schema:
            self.__training_test_schema = json.load(training_test_schema)
        training_test_schema.close()

    def validate(self, json_to_validate: dict):
        json_validator = Draft7Validator(self.__training_test_schema)
        try:
            json_validator.validate(json_to_validate)
            return True
        except exceptions.ValidationError as e:
            print(e)
            return False
