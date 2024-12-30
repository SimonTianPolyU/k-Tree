import numpy as np
from sklearn.metrics import mean_squared_error
from collections import deque
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from random import sample
from sklearn.model_selection import train_test_split
import copy
import pandas as pd


class AkTree:
    def __init__(self, max_depth, min_samples, k):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.k = k
        self.nodes = {}  # Only store depth now
        self.adjacency_list = {}  # Stores the indices of each leaf's adjacent leaves.
        self.node_counter = 0  # Stores the index of each node
        self.F = {}  # Stores the features used in each node.
        self.V = {}  # Stores the directions used in each node.
        self.U = {}  # Stores the thresholds used in each node.
        self.sample_indices = {}  # Stores the sample indices of each node.
        self.X = None
        self.y = None
        self.x_min = None
        self.x_max = None
        self.A = None
        self.root = None
        self.children = {}  # This dictionary will store the child nodes for each node.
        self.mse = {}
        self.UU = None
        self.UL = None

    def get_data(self, node):
        # Returns the data of a node
        indices = self.sample_indices[node]
        return [(self.X[i], self.y[i]) for i in indices]

    def fit(self, X, y):
        self.X = X  # Store the training data.
        self.y = y  # Store the training labels.

        scaler = MinMaxScaler()  # Normalize the training data.
        X_normalized = scaler.fit_transform(X)  # Normalize the training data.

        S_N = list(zip(X, y))  # S_N is the set of all training samples.
        self.root = 0  # The root node is the first node.
        self.nodes[self.root] = {'depth': 0}  # The depth of the root node is 0.
        self.adjacency_list[self.root] = []  # The root node has no adjacent nodes.
        self.node_counter += 1  # The next node will be indexed by 1.
        self.sample_indices[self.root] = list(range(len(X)))  # The root node contains all samples.

        neighbors = NearestNeighbors(n_neighbors=self.k, metric='manhattan').fit(
            X_normalized)  # Using manhattan distance to find k neighbors.
        k_neighbors = neighbors.kneighbors(X_normalized,
                                           return_distance=False)  # Return the indices of k neighbors for each sample.

        y_pred_knn = np.array([np.mean(y[k_neighbor_indices]) for k_neighbor_indices in k_neighbors])
        MSE_root = mean_squared_error(y, y_pred_knn)
        self.mse[self.root] = MSE_root

        omega = np.zeros((len(S_N), len(S_N)))  # Initialize omega matrix.
        for i, neighbor_indices in enumerate(k_neighbors):  # For each sample, set the corresponding k neighbors to 1.
            omega[i, neighbor_indices] = 1  # Set the corresponding k neighbors to 1.

        self.x_min = np.min(X, axis=0).reshape(-1, 1)

        # A is a list of length num_features.
        self.x_max = np.max(X, axis=0).reshape(-1, 1)

        self.A = 1 / (self.x_max - self.x_min)

        self.F = {self.root: []}  # Initialize F, V, and U.
        self.V = {self.root: []}  # F, V, and U are lists of length num_features.
        self.U = {self.root: []}  # F, V, and U are lists of length num_features.

        stack = deque([self.root])  # Initialize stack with the root node.

        while stack:
            m = stack.pop()  # Pop the last node from the stack.

            S_m = self.get_data(m)  # Get the data of the node.

            if self.nodes[m]['depth'] >= self.max_depth or len(
                    S_m) <= 2 * self.min_samples - 1:  # If the node is a leaf node, continue.
                continue

            MSE_m = self.mse[m]

            F_m_temp = copy.deepcopy(self.F[m])  # Get the features used in the node.
            U_m_temp = copy.deepcopy(self.U[m])  # Get the directions used in the node.
            V_m_temp = copy.deepcopy(self.V[m])  # Get the thresholds used in the node.
            L_temp = copy.deepcopy(self.adjacency_list)  # Get the adjacency list.
            x_min_temp = copy.deepcopy(self.x_min)  # Get x_min.
            x_max_temp = copy.deepcopy(self.x_max)  # Get x_max.
            A_temp = copy.deepcopy(self.A)  # Get A.
            sample_indices_temp = copy.deepcopy(self.sample_indices)  # Get the sample indices.

            MSE_opt, MSE_m_left_star, MSE_m_right_star, S_m_left_star, S_m_right_star, \
                F_m_left_star, U_m_left_star, V_m_left_star, \
                F_m_right_star, U_m_right_star, V_m_right_star, \
                omega_star, x_min_star, x_max_star, \
                L_star, A_star, sample_indices_star = self.find_optimal_split(
                m, S_m, F_m_temp, U_m_temp, V_m_temp, omega, L_temp, x_min_temp, x_max_temp, A_temp,
                sample_indices_temp
            )  # Find the optimal split for the node.

            if (MSE_opt != None) and MSE_m - MSE_opt > 0:

                m_left, m_right = self.create_child_nodes(m, F_m_left_star[-1], V_m_left_star[-1],
                                                          S_m)  # Create the child nodes.

                stack.extend([m_left, m_right])  # Add the child nodes to the stack.

                for c, F_star, U_star, V_star, MSE_star in zip([m_left, m_right],
                                                               [F_m_left_star, F_m_right_star],
                                                               [U_m_left_star, U_m_right_star],
                                                               [V_m_left_star, V_m_right_star],
                                                               [MSE_m_left_star, MSE_m_right_star]):
                    self.F[c] = F_star
                    self.V[c] = V_star
                    self.U[c] = U_star
                    self.mse[c] = MSE_star

                self.adjacency_list, omega, self.x_min, self.x_max, self.A, self.sample_indices = copy.deepcopy(
                    L_star), copy.deepcopy(omega_star), copy.deepcopy(x_min_star), copy.deepcopy(
                    x_max_star), copy.deepcopy(A_star), copy.deepcopy(
                    sample_indices_star)  # updating based on new returned values.

        self.UL, self.UU = self.track_feature_bound_determination()

        return self.x_min, self.x_max, self.adjacency_list, self.F, self.V, self.U, self.sample_indices, self.A

    def track_feature_bound_determination(self):
        # Initialize UL and UU matrices
        n_nodes = self.node_counter
        n_features = self.X.shape[1]
        UL = np.ones((n_nodes, n_features))
        UU = np.ones((n_nodes, n_features))

        # Iterate through each node
        for node in range(n_nodes):
            F_m = self.F.get(node, [])

            V_m = self.V.get(node, [])

            # Iterate through each feature
            for feature in range(n_features):
                # Check if the feature is in the list for the current node
                for i in range(len(F_m)):
                    if F_m[i] == feature:
                        if V_m[i] == 1:
                            UL[node, feature] = 0  # Lower bound set
                        elif V_m[i] == -1:
                            UU[node, feature] = 0  # Upper bound set

        return UL, UU

    def find_optimal_split(self, m, S_m, F_m, U_m, V_m, omega, L, x_min, x_max, A, sample_indices):
        MSE_best = float('inf')
        p_star, theta_p_star = None, None
        # The new initialization for best values
        S_star_left, S_star_right = None, None
        F_star_left, U_star_left, V_star_left = [], [], []
        F_star_right, U_star_right, V_star_right = [], [], []
        omega_star, x_min_star, x_max_star, L_star, A_star, samples_indices_star = None, None, None, None, None, None
        # The new initialization for best values

        num_features = len(S_m[0][0])

        # Remove node m from the adjacency lists of its neighbors.
        for neighbor in L[m]:
            L[neighbor].remove(m)

        MSE_best_left = float('inf')
        MSE_best_right = float('inf')

        # Add the new nodes with empty adjacency lists.
        L[len(self.nodes)] = []
        L[len(self.nodes) + 1] = []

        # Remove node m's adjacency list.
        del L[m]

        for p in range(num_features):
            values = [sample[0][p] for sample in S_m]

            unique_values = set(values)

            # If the feature is one-hot encoded
            if unique_values == {0, 1}:
                unique_values = 0.5  # split point between 0 and 1
            else:
                # If not one-hot encoded, use the binning technique
                min_val, max_val = min(values), max(values)
                bin_edges = np.linspace(min_val, max_val, 6)  # creates 5 split points
                # Skip the first and the last edge since they represent the min and max values.
                unique_values = bin_edges[1:-1]

            for theta_p in unique_values:
                S_left, S_right, sample_indices_prime = self.create_child_nodes_temp(m, p, theta_p, S_m,
                                                                                     len(self.nodes), sample_indices)
                # print(p, theta_p)
                if len(S_left) < self.min_samples or len(S_right) < self.min_samples:
                    continue

                omega_prime = None if omega is None else copy.deepcopy(omega)
                x_min_prime = None if x_min is None else copy.deepcopy(x_min)
                x_max_prime = None if x_max is None else copy.deepcopy(x_max)
                A_prime = None if A is None else copy.deepcopy(A)

                for c, S_c in zip([len(self.nodes), len(self.nodes) + 1], [S_left, S_right]):  # m_left and m_right
                    F_c_prime = F_m + [p]
                    U_c_prime = U_m + [theta_p]
                    V_c_prime = V_m + [-1] if c == len(self.nodes) else V_m + [1]
                    x_min_prime, x_max_prime, A_prime = self.update_node_bounds(c, S_c, F_c_prime, U_c_prime, V_c_prime,
                                                                                x_min_prime, x_max_prime, A_prime)

                # Update the adjacency list here.
                L_prime = copy.deepcopy(L)
                L_prime = self.update_adjacency_list(len(self.nodes), L_prime, x_min_prime, x_max_prime)

                y_actual_left, y_pred_left = [], []
                for x, y in S_left:
                    y_hat, _ = self.compute_k_neighbors_and_prediction(len(self.nodes), x, S_left, omega_prime, L_prime,
                                                                       x_min_prime, x_max_prime, A_prime,
                                                                       sample_indices_prime)
                    y_actual_left.append(y)
                    y_pred_left.append(y_hat)

                y_actual_right, y_pred_right = [], []
                for x, y in S_right:
                    y_hat, _ = self.compute_k_neighbors_and_prediction(len(self.nodes) + 1, x, S_right, omega_prime,
                                                                       L_prime, x_min_prime, x_max_prime, A_prime,
                                                                       sample_indices_prime)
                    y_actual_right.append(y)
                    y_pred_right.append(y_hat)

                MSE_current_left = mean_squared_error(y_actual_left, y_pred_left)
                MSE_current_right = mean_squared_error(y_actual_right, y_pred_right)

                # Combine actual values and predictions from both left and right child nodes
                y_actual_combined = y_actual_left + y_actual_right
                y_pred_combined = y_pred_left + y_pred_right

                # Compute the MSE for the current split
                MSE_current = mean_squared_error(y_actual_combined, y_pred_combined)

                if MSE_current < MSE_best:
                    MSE_best = MSE_current
                    MSE_best_left = MSE_current_left
                    MSE_best_right = MSE_current_right
                    p_star, theta_p_star = p, theta_p
                    S_star_left, S_star_right = copy.deepcopy(S_left), copy.deepcopy(S_right)
                    F_star_left, V_star_left, U_star_left = copy.deepcopy(F_m + [p]), copy.deepcopy(
                        V_m + [-1]), copy.deepcopy(U_m + [theta_p])
                    F_star_right, V_star_right, U_star_right = copy.deepcopy(F_m + [p]), copy.deepcopy(
                        V_m + [1]), copy.deepcopy(U_m + [theta_p])
                    omega_star = copy.deepcopy(omega_prime)
                    x_min_star = copy.deepcopy(x_min_prime)
                    x_max_star = copy.deepcopy(x_max_prime)
                    L_star = copy.deepcopy(L_prime)
                    A_star = copy.deepcopy(A_prime)
                    samples_indices_star = copy.deepcopy(sample_indices_prime)

        if p_star is None:
            return None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None

        return MSE_best, MSE_best_left, MSE_best_right, S_star_left, S_star_right, \
            F_star_left, U_star_left, V_star_left, \
            F_star_right, U_star_right, V_star_right, \
            omega_star, x_min_star, x_max_star, \
            L_star, A_star, samples_indices_star

    def create_child_nodes(self, m, p_star, theta_p_star, S_m):
        m_left = self.node_counter  # m_left is the next node to be created
        self.node_counter += 1  # Increment the node counter
        m_right = self.node_counter  # m_right is the next node to be created
        self.node_counter += 1  # Increment the node counter

        # After creating m_left and m_right nodes:
        self.children[m] = (m_left, m_right)

        self.nodes[m_left] = {'depth': self.nodes[m]['depth'] + 1}  # Set the depth of m_left
        self.nodes[m_right] = {'depth': self.nodes[m]['depth'] + 1}  # Set the depth of m_right
        self.adjacency_list[m_left] = []  # m_left has no adjacent nodes
        self.adjacency_list[m_right] = []  # m_right has no adjacent nodes

        S_left_indices = [self.sample_indices[m][i] for i, s in enumerate(S_m) if
                          s[0][p_star] <= theta_p_star]  # Get the indices of samples in S_left
        S_right_indices = [self.sample_indices[m][i] for i, s in enumerate(S_m) if
                           s[0][p_star] > theta_p_star]  # Get the indices of samples in S_right

        self.sample_indices[m_left] = S_left_indices  # Set the indices of samples in S_left
        self.sample_indices[m_right] = S_right_indices  # Set the indices of samples in S_right

        return m_left, m_right  # Return the indices of the child nodes

    # This function is used to create child nodes without updating the adjacency list
    def create_child_nodes_temp(self, m, p_star, theta_p_star, S_m, node_counter, sample_indices):

        S_right = [s for s in S_m if s[0][p_star] > theta_p_star]  # Get the samples in S_right
        S_left = [s for s in S_m if s[0][p_star] <= theta_p_star]  # Get the samples in S_left

        m_left = node_counter  # m_left is the next node to be created
        node_counter += 1  # Increment the node counter
        m_right = node_counter  # m_right is the next node to be created

        S_left_indices = [sample_indices[m][i] for i, s in enumerate(S_m) if
                          s[0][p_star] <= theta_p_star]  # Get the indices of samples in S_left
        S_right_indices = [sample_indices[m][i] for i, s in enumerate(S_m) if
                           s[0][p_star] > theta_p_star]  # Get the indices of samples in S_right

        sample_indices[m_left] = S_left_indices  # Set the indices of samples in S_left
        sample_indices[m_right] = S_right_indices  # Set the indices of samples in S_right

        return S_left, S_right, sample_indices

    # This function is used to update the bounds of a node
    def update_node_bounds(self, m, S_m, F_m, U_m, V_m, x_min, x_max, A):
        # Add a column for node m to x_min, x_max, and A
        num_features = len(S_m[0][0])  # Extracting number of features from a sample tuple in S_m

        inf_min = np.full((num_features, 1), -np.inf)
        inf_max = np.full((num_features, 1), np.inf)

        x_min = np.column_stack((x_min, inf_min))
        x_max = np.column_stack((x_max, inf_max))

        zeros_column = np.zeros((A.shape[0], 1))
        A = np.column_stack((A, zeros_column))

        # Update feature bounds based on samples in S_m
        for p in range(num_features):

            values = [sample[0][p] for sample in S_m]  # Extracting feature p from each sample in S_m

            x_min[p, m] = min(values)
            x_max[p, m] = max(values)
        # Adjust bounds for features used in splits leading to node m
        for i in range(len(F_m)):
            p = F_m[i]
            if V_m[i] == -1:
                x_max[p, m] = U_m[i]
            else:
                x_min[p, m] = U_m[i]
        # Compute normalization factors
        for p in range(num_features):
            if x_max[p, m] == x_min[p, m]:
                A[p, m] = 0
            else:
                A[p, m] = 1 / (x_max[p, m] - x_min[p, m])

        return x_min, x_max, A

    def update_adjacency_list(self, m, L, x_min, x_max):
        for l in L.keys():
            # Skip the current node
            if l != m:
                isAdjacent = True
                # Iterate through all features
                for p in range(x_min.shape[0]):
                    if x_min[p, l] > x_max[p, m] or x_max[p, l] < x_min[p, m]:
                        isAdjacent = False
                        break
                if isAdjacent:
                    L[l].append(m)
                    L[m].append(l)
        return L

    def compute_k_neighbors_and_prediction(self, m, x, S_m, omega, L, x_min, x_max, A, sample_indices):

        def array_in_list(array, list_of_arrays):
            return any((array == item).all() for item in list_of_arrays)

        def get_data_temp(node, sample_indices):
            # Returns the data of a node
            indices = sample_indices[node]
            return [(self.X[i], self.y[i]) for i in indices]

        distances = []

        # Find the closest k neighbors within S_m
        for sample, _ in S_m:
            distances.append(self.compute_distance(x, sample, A, m))

        if len(S_m) >= self.k:
            sorted_indices = np.argsort(distances)[:self.k]
            K_x = [S_m[i] for i in sorted_indices]
            distances = [distances[i] for i in sorted_indices]  # Update distances to store only k neighbors' distances
        else:
            K_x = copy.deepcopy(S_m)

        # Retrieve historical neighbors of x not in S_m using omega
        H_x = []
        x_index_in_self_X = np.where((self.X == x).all(axis=1))[0][0]

        for i, omega_value in enumerate(omega[x_index_in_self_X]):
            if omega_value == 1 and not array_in_list(self.X[i], [sample[0] for sample in S_m]):
                H_x.append((self.X[i], self.y[i]))

        # Check if each x' in H_x is in an adjacent node m_prime of m
        for x_prime, y_prime in H_x:
            for m_prime in L[m]:
                data_m_prime_samples = {tuple(sample[0]) for sample in get_data_temp(m_prime, sample_indices)}
                if tuple(x_prime) in data_m_prime_samples:
                    distances.append(self.compute_distance_adjacent(x, x_prime, m, m_prime, A, x_min, x_max))
                    K_x.append((x_prime, y_prime))

        # If we have more than k neighbors, we should reduce them to k by considering the closest ones
        if len(K_x) > self.k:
            sorted_indices = np.argsort(distances)[:self.k]
            K_x = [K_x[i] for i in sorted_indices]

        # Compute prediction (assuming regression task) as the average of y-values of neighbors
        y_hat = np.mean([sample[1] for sample in K_x])

        # Update omega for the new neighboring relationships
        omega_updated = copy.deepcopy(omega)
        omega_updated[x_index_in_self_X, :] = 0  # Reset the row corresponding to x
        omega_updated[:, x_index_in_self_X] = 0  # Reset the column corresponding to x

        for sample, _ in K_x:
            sample_index_in_self_X = np.where((self.X == sample).all(axis=1))[0][0]
            omega_updated[x_index_in_self_X, sample_index_in_self_X] = 1  # Update the row of omega
            omega_updated[sample_index_in_self_X, x_index_in_self_X] = 1  # Update the column of omega

        return y_hat, omega_updated

    def compute_distance(self, x1, x2, A, m):
        # Using Eq12, compute the Manhattan distance considering normalization and bounds
        normalized_diff = [abs(x1i - x2i) * Ai for x1i, x2i, Ai in zip(x1, x2, A[:, m])]
        # Compute the Manhattan distance considering bounds only
        return sum(diff for diff in normalized_diff)

    def compute_distance_adjacent(self, x1, x2, m1, m2, A, x_min, x_max):
        total_distance = 0

        for p in range(len(x1)):
            x_min_overlap = max(x_min[p, m1], x_min[p, m2])
            x_max_overlap = min(x_max[p, m1], x_max[p, m2])
            f_p_star = self.compute_optimal_solution(
                x1[p], x2[p], A[p, m1], A[p, m2], x_min_overlap, x_max_overlap
            )
            total_distance += f_p_star

        return total_distance

    def compute_optimal_solution(self, x_0_p, x_0_prime_p, a_l0_p, a_l0_prime_p, x_min_overlap_p, x_max_overlap_p):
        # (1)
        if x_0_p < x_0_prime_p and a_l0_p < a_l0_prime_p:
            if x_0_prime_p >= x_max_overlap_p >= x_0_p:
                return a_l0_p * (x_max_overlap_p - x_0_p) + a_l0_prime_p * (x_0_prime_p - x_max_overlap_p)
            elif x_min_overlap_p <= x_0_prime_p < x_max_overlap_p:
                return a_l0_p * (x_0_prime_p - x_0_p)

        # (2)
        if x_0_p < x_0_prime_p and a_l0_p >= a_l0_prime_p:
            if x_min_overlap_p < x_0_p <= x_max_overlap_p:
                return a_l0_prime_p * (x_0_prime_p - x_0_p)
            elif x_0_p <= x_min_overlap_p <= x_0_prime_p:
                return a_l0_p * (x_min_overlap_p - x_0_p) + a_l0_prime_p * (x_0_prime_p - x_min_overlap_p)

        # (3)
        if x_0_p >= x_0_prime_p and a_l0_p < a_l0_prime_p:
            if x_min_overlap_p < x_0_prime_p <= x_max_overlap_p:
                return a_l0_p * (x_0_p - x_0_prime_p)
            elif x_0_prime_p <= x_min_overlap_p <= x_0_p:
                return a_l0_p * (x_0_p - x_min_overlap_p) + a_l0_prime_p * (x_min_overlap_p - x_0_prime_p)

        # (4)
        if x_0_p >= x_0_prime_p and a_l0_p >= a_l0_prime_p:
            if x_0_p >= x_max_overlap_p >= x_0_prime_p:
                return a_l0_p * (x_0_p - x_max_overlap_p) + a_l0_prime_p * (x_max_overlap_p - x_0_prime_p)
            elif x_min_overlap_p <= x_0_p < x_max_overlap_p:
                return a_l0_prime_p * (x_0_p - x_0_prime_p)

        return None

    def get_children(self, node):
        return self.children.get(node, (None, None))

    def determine_leaf(self, test_sample):
        current_node = 0  # Start from the root
        while True:
            children = self.get_children(current_node)

            if not children or children == (None, None):
                return current_node

            # Ensure that children[0] is not None
            if children[0] is None:
                raise ValueError(f"Invalid child node for parent node {current_node}.")

            current_node = children[0]

            feature_index = self.F[current_node][-1]  # Get feature for current node
            threshold = self.U[current_node][-1]  # Get threshold for that feature

            # Based on the test sample's value for the feature, choose the next node
            if test_sample[feature_index] <= threshold:
                current_node = children[0]  # Left child
            else:
                current_node = children[1]  # Right child

    def predict_all(self, X_test):
        """
        Predicts for an entire test set.
        """
        predictions = []
        proportions_adjacent = []
        for x0 in X_test:
            avg_prediction, proportion_adjacent = self.predict(x0)
            predictions.append(avg_prediction)
            proportions_adjacent.append(proportion_adjacent)

        mean_proportions_adjacent = np.mean(proportions_adjacent)
        return predictions, mean_proportions_adjacent

    def predict(self, x0):
        # Determine the leaf l_0 that houses x0
        l_0 = self.determine_leaf(x0)

        # Adjust x_min and x_max to include x0 if necessary
        for p in range(len(x0)):
            if self.UL[l_0, p] == 1 and x0[p] < self.x_min[p, l_0]:
                # Adjust the min and max values
                self.x_min[p, l_0] = x0[p]
                self.A[p, l_0] = 1.0 / (self.x_max[p, l_0] - self.x_min[p, l_0])
            elif self.UU[l_0, p] == 1 and x0[p] > self.x_max[p, l_0]:
                self.x_max[p, l_0] = x0[p]
                self.A[p, l_0] = 1.0 / (self.x_max[p, l_0] - self.x_min[p, l_0])

        distances = []
        is_adjacent = []

        # For training sample x_i in node l_0, compute the Manhattan distance
        for x, y in self.get_data(l_0):
            distances.append((self.compute_distance(x0, x, self.A, l_0), y))
            is_adjacent.append(False)

        # For nodes l'_0 adjacent to l_0
        for l_prime in self.adjacency_list[l_0]:
            for x, y in self.get_data(l_prime):
                # Obtain the minimum value f_p^* for each feature p (use a predefined function)
                # Compute the minimum distance (use a predefined function for equation (20))
                distances.append(
                    (self.compute_distance_adjacent(x0, x, l_0, l_prime, self.A, self.x_min, self.x_max), y))
                is_adjacent.append(True)

        # Sort by distance and select k samples
        k_samples = sorted(zip(distances, is_adjacent), key=lambda x: x[0])[:self.k]

        adjacent_count = sum(1 for _, is_adj in k_samples if is_adj)

        avg_prediction = sum(yi for ((_, yi), _) in k_samples) / len(k_samples)

        proportion_adjacent = adjacent_count / self.k

        # Return the average y-value of the selected samples as the prediction
        return avg_prediction, proportion_adjacent
