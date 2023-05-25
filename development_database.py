import shelve


class DevelopmentDatabase:

    def __init__(self, db_name: str):
        self.__db = shelve.open(db_name)

    def close(self):
        self.__db.close()

    def store_json(self, json_to_store: dict):
        self.__db['set'] = json_to_store

    def get_json(self):
        return self.__db['set']

    def del_json(self):
        del self.__db['set']

    def store_list(self, list_to_store: list):
        self.__db['top_5'] = list_to_store

    def get_list(self):
        try:
            return self.__db['top_5']
        except KeyError:
            return None

    def del_list(self):
        del self.__db['top_5']
