import distance
import math
import numpy as np


def iterate_batches(points, batch_size):
    samples_number = math.ceil(points.shape[0] / batch_size)
    for sample_id in range(0, samples_number):
        sample = points[sample_id * batch_size: (sample_id + 1) * batch_size]
        yield sample

def distance_function(sample, dist, dist_metric):
    try:
        dist_mean = np.mean(dist, axis = 0)
        dist_mean_list = [dist_mean]*sample.shape[0]
        return dist_metric(sample, dist_mean_list)
    except:
        print("Failure")
        exit()

class ModularDetector:
    def __init__(self, threshold=3, new_dist_buffer_size=3, batch_size=3, max_dist_size=0, dist_metric = distance.euclidean, DEBUG=False):
        self.threshold_ratio = threshold
        self.max_dist_size = max_dist_size
        self.new_dist_buffer_size = new_dist_buffer_size
        self.batch_size = batch_size
        self.is_creating_new_dist = True
        self.dist = []
        self.dist_values = []
        self.locations = []
        self.dist_metric = dist_metric
        self.DEBUG=DEBUG
        if self.DEBUG:
            self.values = []

    def detect(self, data):
        for batch_id, batch in enumerate(iterate_batches(data, batch_size=self.batch_size)):
            if self.is_creating_new_dist:
                self.dist.extend(batch)

                if len(self.dist) >= self.new_dist_buffer_size:
                    self.is_creating_new_dist = False
                    values = [distance_function(np.array(s), np.array(self.dist), self.dist_metric) for s in
                                        iterate_batches(np.array(self.dist), self.batch_size)]
                    self.threshold = np.max(values) * self.threshold_ratio

            else:
                value = distance_function(np.array(batch), np.array(self.dist), self.dist_metric)

                if value > self.threshold:
                    self.locations.append(batch_id * self.batch_size)
                    self.dist = []
                    self.is_creating_new_dist = True

                if self.max_dist_size == 0 or len(self.dist) < self.max_dist_size:
                    self.dist.extend(batch)
                    values = [distance_function(np.array(s), np.array(self.dist), self.dist_metric) for s in
                                        iterate_batches(np.array(self.dist), self.batch_size)]
                    self.threshold = np.max(values) * self.threshold_ratio

        return self.locations

