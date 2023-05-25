from concurrent.futures.thread import ThreadPoolExecutor

import requests
from flask import Flask, request


class RestServer(Flask):

    def __init__(self, *args, **kwargs):
        self.__thread_pool = ThreadPoolExecutor(max_workers=1)
        self.__development_system = kwargs.pop("develop_sys")
        super(RestServer, self).__init__(*args, **kwargs)
        self.add_url_rule('/send_json', 'receive_json', self.receive_json, methods=['POST'])

    def receive_json(self):
        received_json = request.get_json()
        self.__thread_pool.submit(self.__development_system.first_training_mode, received_json)
        return 'Success'

    @staticmethod
    def send_file(file_name: str, ip: str, port: int, resource: str):
        url = "http://" + ip + ":" + str(port) + "/" + resource
        with open(file_name, 'rb') as f:
            files = {'file': f}
            response = requests.post(url, files=files)
        if response.status_code != 200:
            return False
        return True
