from abc import ABC
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

@dataclass
class Task:
    name: str
    data: np.array
    labels: np.array


class ClusteredDataReader(ABC):
    def __init__(self, data_path):
        data = np.load(data_path, allow_pickle=True)
        self.train_tasks = []
        self.test_tasks = []

        for c in data:
            self.train_tasks.append(Task(name=c['name'], data=c['train_data'], labels=None))
            if 'test_data' in c and len(c['test_data']) > 0:
                test_data = c['test_data']
                test_labels = c['test_labels']
                self.test_tasks.append(Task(name=c['name'], data=test_data, labels=test_labels))

    def load_test_tasks(self) -> List[Task]:
        return self.test_tasks

    def iterate_tasks(self) -> Iterable[Task]:
        return self.train_tasks
