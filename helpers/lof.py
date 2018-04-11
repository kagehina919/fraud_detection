import pandas as pd
import numpy as np


class LocalOutlierFactor:
    """
    """
    def __init__(self, df, k, num_outliers):
        """
        """
        self.k = k
        self.num_outliers = num_outliers
        self.points = df.values
        self.num_points = len(self.points)
        self.k_distance = {}
        self.k_neighbors = {}
        self.distances = np.zeros((self.num_points, self.num_points,))
        self.reach_distance = np.zeros((self.num_points, self.num_points,))
        self.local_outlier_factor = np.zeros(self.num_points)
        self.calculate_distances()
        self.compute_k_distances()
        self.compute_local_outlier_factor()
        self.find_outliers()

    def calculate_distances(self):
        """
        :param points:
        :return:
        """
        for idx_a, point_a in enumerate(self.points):
            for idx_b, point_b in enumerate(self.points):
                self.distances[idx_a][idx_b] = np.linalg.norm(point_a-point_b)

    def compute_k_distances(self):
        """

        :return:
        """
        for i in range(self.num_points):
            distances = self.distances[i]
            k_dist = np.partition(distances, self.k-1)[self.k-1]
            sort_index = np.argsort(distances)

            neighbors = []
            for idx, dist in enumerate(distances):
                if dist <= k_dist:
                    neighbors.append(sort_index[idx])

            self.k_distance[i] = k_dist
            self.k_neighbors[i] = neighbors

    def compute_reach_distance(self):
        """

        :return:
        """
        for idx_a, point_a in enumerate(self.points):
            for idx_b, point_b in enumerate(self.points):
                self.reach_distance[idx_a][idx_b] = max(self.k_distance[idx_a], self.distances[idx_a][idx_b])

    def compute_local_outlier_factor(self):
        """

        :return:
        """
        self.compute_reach_distance()

        for i in range(self.num_points):
            sum_reach_distance = 0
            for point in self.k_neighbors[i]:
                sum_reach_distance += self.reach_distance[point][i]
            self.local_outlier_factor[i] = len(self.k_neighbors[i])/sum_reach_distance

    def find_outliers(self):
        """

        :return:
        """
        sorted_outlier_factor_indexes = np.argsort(self.local_outlier_factor)
        print(sorted_outlier_factor_indexes[-self.num_outliers:])
