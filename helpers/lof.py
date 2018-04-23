import numpy as np
import pickle

class LocalOutlierFactor:
    """
    """
    def __init__(self, df):
        """
        """
        self.points = df.values
        self.num_points = len(self.points)
        self.k_distance = {}
        self.k_neighbors = {}
        self.distances = np.zeros((self.num_points, self.num_points,))
        self.reach_distance = np.zeros((self.num_points, self.num_points,))
        self.local_outlier_factor = np.zeros(self.num_points)

    def mahanalobis_distance(self, a, b):
        """
        Computes the Mahalanobis distance between two 1-D arrays.
        The Mahalanobis distance between 1-D arrays `u` and `v`, is defined as
        .. math::
           \\sqrt{ (u-v) V^{-1} (u-v)^T }
        where ``V`` is the covariance matrix.  Note that the argument `VI`
        is the inverse of ``V``.
        Parameters
        ----------
        u : (N,) array_like
            Input array.
        v : (N,) array_like
            Input array.
        VI : ndarray
            The inverse of the covariance matrix.
        Returns
        -------
        mahalanobis : double
            The Mahalanobis distance between vectors `u` and `v`.
        """
        X = np.array([a, b]).T
        cov_X = np.cov(X)
        delta = a - b
        VI = np.linalg.pinv(cov_X)
        m = np.dot(np.dot(delta, VI), delta)
        return np.sqrt(m)

    def calculate_distances(self, reload=False):
        """
        :param points:
        :return:
        """
        if reload:
            for idx_a, point_a in enumerate(self.points):
                for idx_b, point_b in enumerate(self.points):
                    self.distances[idx_a][idx_b] = self.mahanalobis_distance(point_a, point_b)

            with open('distances.pkl', 'wb') as pkl:
                pickle.dump(self.distances, pkl)
        else:
            with open('distances.pkl', 'rb') as pkl:
                self.distances = pickle.load(pkl)

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

    def find_outliers(self, k, num_outliers):
        """

        :return:
        """
        self.k = k
        self.num_outliers = num_outliers
        self.calculate_distances(reload=False)
        self.compute_k_distances()
        self.compute_local_outlier_factor()
        sorted_outlier_factor_indexes = np.argsort(self.local_outlier_factor)
        # print(sorted_outlier_factor_indexes[-self.num_outliers:])
        return sorted_outlier_factor_indexes[-self.num_outliers:]