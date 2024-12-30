import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.utils.validation import check_X_y, check_array


class PkTree:
    def __init__(self, dt_params=None, k=None):
        if dt_params is None:
            self.dt_params = {}  # Decision Tree parameters
        else:
            self.dt_params = dt_params
        self.k = k  # the new hyperparameter, similar to k in kNN

    # Add get_params and set_params methods
    def get_params(self, deep=True):
        return {'dt_params': self.dt_params, 'k': self.k}

    def set_params(self, **params):
        if 'dt_params' in params:
            self.dt_params = params['dt_params']
        if 'k' in params:
            self.k = params['k']
        else:
            dt_params = {}
            for parameter, value in params.items():
                if parameter.startswith('dt_params__'):
                    _, param = parameter.split('__', 1)
                    dt_params[param] = value
            self.dt_params = dt_params
        return self

    def train_decision_tree(self, X, y):
        # Train a Decision Tree on the input data
        self.dt = DecisionTreeRegressor(**self.dt_params)
        self.dt.fit(X, y)
        return self.dt

    def map_samples_to_nodes(self, X):
        leaf_indices = self.dt.apply(X)
        unique_nodes, inverse_indices = np.unique(leaf_indices, return_inverse=True)
        return {node: np.where(inverse_indices == i)[0] for i, node in enumerate(unique_nodes)}

    def compute_a(self):
        # Compute the a matrix for each node
        self.a = {}
        for node, space in self.node_space_info.items():
            diff = space['upper_bound'] - space['lower_bound']  # Calculate the difference
            self.a[node] = np.zeros_like(diff)
            non_zero_mask = diff != 0
            self.a[node][non_zero_mask] = 1.0 / diff[non_zero_mask]

    def determine_node_adjacency_overlap(self, node_space_info):
        adjacency = {}
        overlap_spaces = {}
        for node1, space1 in node_space_info.items():
            for node2, space2 in node_space_info.items():
                if node1 != node2:
                    intersect = True
                    for p in range(len(space1['lower_bound'])):
                        if space1['lower_bound'][p] > space2['upper_bound'][p] or space2['lower_bound'][p] > \
                                space1['upper_bound'][p]:
                            intersect = False
                            break

                    if intersect:
                        adjacency.setdefault(node1, []).append(node2)
                        overlap = {}
                        for p in range(len(space1['lower_bound'])):
                            overlap[p] = (max(space1['lower_bound'][p], space2['lower_bound'][p]),
                                          min(space1['upper_bound'][p], space2['upper_bound'][p]))
                        overlap_spaces[(node1, node2)] = overlap
                        overlap_spaces[(node2, node1)] = overlap

        return adjacency, overlap_spaces

    def update_overlap_spaces_for_node(self, node):
        for adj_node in self.adjacency.get(node, []):
            overlap = {}
            for p in range(len(self.node_space_info[node]['lower_bound'])):
                overlap[p] = (
                    max(self.node_space_info[node]['lower_bound'][p], self.node_space_info[adj_node]['lower_bound'][p]),
                    min(self.node_space_info[node]['upper_bound'][p], self.node_space_info[adj_node]['upper_bound'][p])
                )
            # Update overlap spaces for both node combinations
            self.overlap_spaces[(node, adj_node)] = overlap
            self.overlap_spaces[(adj_node, node)] = overlap

    def get_node_space_info(self):
        dt = self.dt.tree_
        node_space_info = {}
        feature_lists = {}  # To store F_m for each node m
        value_lists = {}  # To store V_m for each node m

        # Initialize the bounds
        n_features = self.X_train.shape[1]
        n_nodes = dt.node_count
        x_min = np.full((n_features, n_nodes), -np.inf)
        x_max = np.full((n_features, n_nodes), np.inf)
        F = {node: [] for node in range(n_nodes)}
        V = {node: [] for node in range(n_nodes)}

        def dfs(node, F_m, V_m):
            if dt.children_left[node] != -1:  # if not a leaf node
                feature = dt.feature[node]
                threshold = dt.threshold[node]
                left_child = dt.children_left[node]
                right_child = dt.children_right[node]

                # Update bounds and lists for children
                x_max[feature, left_child] = threshold
                x_min[feature, right_child] = threshold
                F_left = F_m + [feature]
                F_right = F_m + [feature]
                V_left = V_m + [-1]
                V_right = V_m + [1]

                # Recursive calls for child nodes
                dfs(left_child, F_left, V_left)
                dfs(right_child, F_right, V_right)

            else:  # if a leaf node
                samples_index = self.node_to_samples[node]
                samples = self.X_train[samples_index]

                for feature in range(n_features):
                    if feature in F_m:
                        if x_min[feature, node] == -np.inf:
                            x_min[feature, node] = samples[:, feature].min()
                        if x_max[feature, node] == np.inf:
                            x_max[feature, node] = samples[:, feature].max()
                    else:
                        x_min[feature, node] = samples[:, feature].min()
                        x_max[feature, node] = samples[:, feature].max()

                # Store information in node_space_info
                node_space_info[node] = {
                    'upper_bound': x_max[:, node],
                    'lower_bound': x_min[:, node]
                }

            # Store F_m and V_m for each node
            feature_lists[node] = F_m
            value_lists[node] = V_m

        # Start the DFS from the root node
        dfs(0, F[0], V[0])

        return node_space_info, feature_lists, value_lists

    def track_feature_bound_determination(self, feature_lists, value_lists):
        # Initialize UL and UU matrices
        dt = self.dt.tree_
        n_nodes = dt.node_count
        n_features = self.X_train.shape[1]
        UL = np.ones((n_nodes, n_features))
        UU = np.ones((n_nodes, n_features))

        # Iterate through each node
        for node in range(n_nodes):
            F_m = feature_lists.get(node, [])
            V_m = value_lists.get(node, [])

            # Iterate through each feature
            for feature in range(n_features):
                # Check if the feature is in the list for the current node
                for i in range(len(F_m)):
                    if F_m[i] == feature:
                        if V_m[i] == 1:
                            UL[node, feature] = 0  # Lower bound set; for right node
                        elif V_m[i] == -1:
                            UU[node, feature] = 0  # Upper bound set; for left node

        return UL, UU

    def fit(self, X, y):
        self.X_train, self.y_train = check_X_y(X, y)
        self.train_decision_tree(self.X_train, self.y_train)
        self.node_to_samples = self.map_samples_to_nodes(self.X_train)
        self.node_space_info, self.feature_lists, self.value_lists = self.get_node_space_info()
        self.UL, self.UU = self.track_feature_bound_determination(self.feature_lists, self.value_lists)
        self.compute_a()
        self.adjacency, self.overlap_spaces = self.determine_node_adjacency_overlap(self.node_space_info)
        # If the new hyperparameter isn't specified, use the one from the Decision Tree
        if self.k is None:
            self.k = self.dt.min_samples_leaf
            self.k = min(self.k, min(len(samples) for samples in self.node_to_samples.values()))
        return self

    def compute_f_star(self, x, x_prime, a, a_prime, x_min, x_max):
        #     """Compute the f* value based on conditions in Proposition 2."""
        # (1)
        f_star = None

        if x < x_prime and a < a_prime:
            if x_prime >= x_max >= x:
                return a * (x_max - x) + a_prime * (x_prime - x_max)
            elif x_min <= x_prime < x_max:
                return a * (x_prime - x)
        # (2)
        if x < x_prime and a >= a_prime:
            if x_min < x <= x_max:
                return a_prime * (x_prime - x)
            elif x <= x_min <= x_prime:
                return a * (x_min - x) + a_prime * (x_prime - x_min)
        # (3)
        if x >= x_prime and a < a_prime:
            if x_min < x_prime <= x_max:
                return a * (x - x_prime)
            elif x_prime <= x_min <= x:
                return a * (x - x_min) + a_prime * (x_min - x_prime)

        # (4)
        if x >= x_prime and a >= a_prime:
            if x >= x_max >= x_prime:
                return a * (x - x_max) + a_prime * (x_max - x_prime)
            elif x_min <= x < x_max:
                return a_prime * (x - x_prime)

        return None if f_star is None else f_star

    def predict(self, X):
        X = check_array(X)
        preds = []
        proportions_adjacent_neighbors = []
        for x in X:

            node = self.dt.apply([x])[0]

            boundary_updated = False

            for p, x_p in enumerate(x):  # loop through each feature p of x_0
                if self.UL[node, p] == 1 and x_p < self.node_space_info[node]['lower_bound'][p]:
                    # Adjust the min and max values
                    self.node_space_info[node]['lower_bound'][p] = min(self.node_space_info[node]['lower_bound'][p],
                                                                       x_p)
                    self.a[node][p] = 1.0 / (self.node_space_info[node]['upper_bound'][p] -
                                             self.node_space_info[node]['lower_bound'][p])
                    boundary_updated = True
                elif self.UU[node, p] == 1 and x_p > self.node_space_info[node]['upper_bound'][p]:
                    self.node_space_info[node]['upper_bound'][p] = max(self.node_space_info[node]['upper_bound'][p],
                                                                       x_p)
                    self.a[node][p] = 1.0 / (self.node_space_info[node]['upper_bound'][p] -
                                             self.node_space_info[node]['lower_bound'][p])
                    boundary_updated = True
                    # Update the a matrix for the node

            if boundary_updated:
                self.update_overlap_spaces_for_node(node)

            same_node_samples = self.node_to_samples.get(node, [])
            same_node_distances = np.sum(self.a[node] * np.abs(x - self.X_train[same_node_samples]), axis=1)
            same_node_distances = np.array(same_node_distances)

            adjacent_nodes = self.adjacency.get(node, [])
            adjacent_samples = [sample for adj_node in adjacent_nodes for sample in
                                self.node_to_samples.get(adj_node, [])]
            adjacent_node_distances = []
            for adj_node in adjacent_nodes:
                for sample in self.node_to_samples.get(adj_node, []):

                    total_distance = 0
                    for p in range(len(self.X_train[0])):
                        x_0 = x[p]
                        x_prime = self.X_train[sample][p]
                        a_0 = self.a[node][p]
                        a_prime = self.a[adj_node][p]
                        overlap = self.overlap_spaces[(node, adj_node) if node < adj_node else (adj_node, node)][p]
                        x_min = overlap[0]
                        x_max = overlap[1]
                        f_star = self.compute_f_star(x_0, x_prime, a_0, a_prime, x_min, x_max)
                        if f_star is not None:
                            total_distance += f_star
                    adjacent_node_distances.append(total_distance)
            adjacent_node_distances = np.array(adjacent_node_distances)

            if len(same_node_samples) > 0 and len(adjacent_samples) > 0:
                all_distances = np.concatenate([same_node_distances, adjacent_node_distances])
                all_samples = np.concatenate([same_node_samples, adjacent_samples])
            elif len(same_node_samples) > 0:
                all_distances = same_node_distances
                all_samples = same_node_samples
            elif len(adjacent_samples) > 0:
                all_distances = adjacent_node_distances
                all_samples = adjacent_samples
            else:
                preds.append(None)
                continue

            k = min(self.k, len(all_samples))  # Ensure k doesn't exceed available samples
            nearest_neighbors = np.argsort(all_distances)[:k]

            same_node_neighbors = [i for i in nearest_neighbors if i < len(same_node_distances)]
            adjacent_node_neighbors = [i - len(same_node_distances) for i in nearest_neighbors if
                                       i >= len(same_node_distances)]

            nearest_same_node_samples = [same_node_samples[i] for i in same_node_neighbors]
            nearest_adjacent_samples = [adjacent_samples[i] for i in adjacent_node_neighbors]

            total_neighbors = len(nearest_same_node_samples) + len(nearest_adjacent_samples)
            if total_neighbors > 0:
                proportion_adjacent = len(nearest_adjacent_samples) / total_neighbors
            else:
                proportion_adjacent = None  # In case there are no neighbors

            proportions_adjacent_neighbors.append(proportion_adjacent)

            nearest_samples = nearest_same_node_samples + nearest_adjacent_samples

            pred = np.mean(self.y_train[nearest_samples])
            preds.append(pred)

        self.effectiveness = np.mean(proportions_adjacent_neighbors)

        return np.array(preds), self.effectiveness