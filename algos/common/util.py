import os
import json
from datetime import datetime


class EpochRecorder:
    def __init__(self, rank, nice_name):
        self.rank = rank
        self.name = '_'.join(nice_name.split(' '))
        self.data = {'epoch': []}
        self.epoch = 1

    def store(self, value):
        if self.data.get(self.name) is None:
            self.data[self.name] = [value]
        else:
            self.data[self.name].append(value)

        self.data['epoch'].append(self.epoch)
        self.epoch += 1

    def dump(self, custom_data=None):
        timestamp = datetime.now().strftime("%H:%M:%S")
        if custom_data is not None:
            for k, v in custom_data.items():
                self.data[k] = v

        filename = f'results/{timestamp}_{self.name}.json'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as handle:
            json.dump(self.data, handle)
