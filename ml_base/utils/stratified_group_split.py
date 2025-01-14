"""
This code is based on the repository https://github.com/joaofig/strat-group-split/blob/main/group_split.py
and on the following article published by João Paulo Figueira on Medium:
https://towardsdatascience.com/stratified-splitting-of-grouped-datasets-using-optimization-bdc12fb6e691
"""
import numpy as np
import pandas as pd

from numpy.random import default_rng
from collections import namedtuple

Solution = namedtuple("Solution", "cost index")

def index_to_str(idx):
    """
    Generates a string representation from an index array.
    @param idx: The NumPy boolean index array.
    @return: The string representation of the array.
    """
    num_chars = int(idx.shape[0] / 6 + 0.5)
    s = ""
    for i in range(num_chars):
        b = i * 6
        six = idx[b:b+6]
        c = 0
        for j in range(six.shape[0]):
            c = c * 2 + int(six[j])
        s = s + chr(c+32)
    return s


def create_initial_solution(sample_counts, p, random_state):
    """
    Creates the initial solution using a random shuffle
    @param sample_counts: The problem array.
    @param p: The validation set proportion.
    @return: A solution array.
    """
    rng = default_rng(seed=random_state)
    group_count = sample_counts.shape[0]
    idx = np.zeros(group_count, dtype=bool)
    rnd_idx = rng.permutation(group_count)
    start_count = 0
    sample_size = sample_counts[:, 0].sum()
    for i in range(group_count):
        start_count += sample_counts[rnd_idx[i], 0]
        idx[rnd_idx[i]] = True
        if start_count > sample_size * p:
            break
    return idx


def calculate_cost(sample_counts, idx, p, use_norm_cost=False):
    """
    Calculates the cost of a given solution
    @param sample_counts: The problem array.
    @param idx: The solution array to evaluate.
    @param p: The train/validation split proportion.
    @return: The cost value.
    """
    valid_count_prop = sample_counts[idx, 0].sum()  # Num. de muestras totales propuesto en la partición de validación  (10,960 en el ejemplo)
    total_count = sample_counts[:, 0].sum()  # Num. de muestras totales (36,848 en el ejemplo)
    valid_perc_prop = valid_count_prop / total_count  # % de muestras en validación propuesto (29.74% en el ejemplo)

    if use_norm_cost:
        # Qué tan lejos está el porcentaje de muestras de validación propuesto, del que debería tener
        # (% de muestras propuesto en validación - % de muestras que debería de haber en validación) ** 2
        cost = (valid_perc_prop - p) ** 2  #  (.2974 - .3)
    else:
        # Qué tan lejos está el número de muestras de validación propuesto, del que debería tener
        # (Num. muestras propuesto en validación - Num. de muestras que debería de haber en validación) ** 2
        cost = (valid_count_prop - total_count * p) ** 2  # (10,960 - 11,054.4)

    for i in range(1, sample_counts.shape[1]):
        # % de muestras de la Clase-i dentro del total   (20.70% de Class 1 en el ejemplo)
        r = sample_counts[:, i].sum() / total_count
        if use_norm_cost:
            # Qué tan lejos está la Clase-i de su proporción de muestras dentro del total, en la partición de validación propuesta
            # (% de muestras en valid de la Clase-i que hay con esta propuesta  - % de muestras en valid de la Clase-i que debería de haber con esta propuesta) ** 2
            cost += (sample_counts[idx, i].sum() / valid_count_prop - r) ** 2  #  (2,328 / 10,960 - .2070 == .2124 - .2070)
        else:
            # Con esta propuesta, qué tan lejos está la Clase-i de su proporción de muestras en valid respecto de su proporción de muestras en el total
            # (Num. de muestras en valid que hay de la Clase-i con esta propuesta  - Num. de muestras en valid de la Clase-i que debería de haber con esta propuesta) ** 2
            # cost += (sample_counts[idx, i].sum() - r * valid_count_prop) ** 2  #  (2,328 - .207 * 10,960)
            cost += (sample_counts[idx, i].sum() - sample_counts[:, i].sum() * valid_perc_prop) ** 2  #  (2,328 - 7,626 * .2974 == 2,328 - 2,267.97)
    return cost / 2.0


def calculate_cost_grad(sample_counts, idx, p, use_norm_cost=False):
    """
    Calculates the cost gradient of a given solution
    @param sample_counts: The problem array.
    @param idx: The solution array to evaluate.
    @param p: The train/validation split proportion.
    @return: The cost value.
    """
    grad = np.zeros(sample_counts.shape[1])

    total_count = sample_counts[:, 0].sum()  # Num. de muestras totales (36,848 en el ejemplo)
    valid_count_prop = sample_counts[idx, 0].sum()  # Num. de muestras totales propuesto en la partición de validación  (10,960 en el ejemplo)
    valid_perc_prop = valid_count_prop / total_count  # % de muestras en validación propuesto (29.74% en el ejemplo)

    if use_norm_cost:
        # Qué tan lejos está el porcentaje de muestras de validación propuesto, del que debería tener
        # (% de muestras propuesto en validación - % de muestras que debería de haber en validación) ** 2
        grad[0] = p - valid_perc_prop  #  (.3 - .2974)
    else:
        # Qué tan lejos está el número de muestras de validación propuesto, del que debería tener
        # (Num. muestras propuesto en validación - Num. de muestras que debería de haber en validación) ** 2
        grad[0] = total_count * p - valid_count_prop  # (11,054.4 - 10,960)

    for i in range(1, sample_counts.shape[1]):
        # % de muestras de la Clase-i dentro del total   (20.70% de Class 1 en el ejemplo)
        r = sample_counts[:, i].sum() / total_count
        if use_norm_cost:
            # Qué tan lejos está la Clase-i de su proporción de muestras dentro del total, en la partición de validación propuesta
            # (% de muestras en valid de la Clase-i que hay con esta propuesta  - % de muestras en valid de la Clase-i que debería de haber con esta propuesta) ** 2
            grad[i] = r - sample_counts[idx, i].sum() / valid_count_prop  #  ( .2070 - 2,328 / 10,960 ==  .2070 - .2124)
        else:
            # Con esta propuesta, qué tan lejos está la Clase-i de su proporción de muestras en valid respecto de su proporción de muestras en el total
            # (Num. de muestras en valid que hay de la Clase-i con esta propuesta  - Num. de muestras en valid de la Clase-i que debería de haber con esta propuesta) ** 2
            grad[i] = r * valid_count_prop - sample_counts[idx, i].sum()  #  (7,626 * .2974 - 2,328 == 2,267.97 - 2,328)
    return grad


def cosine_similarity(sample_counts, idx, cost_grad):
    """
    Calculates the cosine similarity vector between the problem array
    and the cost gradient vector
    @param sample_counts: The problem array.
    @param idx: The solution vector.
    @param cost_grad: The cost gradient vector.
    @return: The cosine similarity vector.
    """
    c = np.copy(sample_counts)
    c[idx] = -c[idx]            # Reverse direction of validation vectors
    a = np.inner(c, cost_grad)
    b = np.multiply(np.linalg.norm(c, axis=1), np.linalg.norm(cost_grad))
    return np.divide(a, b)


def generate_cosine_move(sample_counts, idx, p, expanded_set, intensify):
    """
    Generates a new move using the cosine similarity.
    @param sample_counts: The problem array.
    @param idx: The solution vector.
    @param p: The validation set proportion.
    @param expanded_set: The set of expanded solutions.
    @param intensify: Intensification / diversification flag.
    @return: The new solution vector.
    """
    cost_grad = calculate_cost_grad(sample_counts, idx, p)
    similarity = cosine_similarity(sample_counts, idx, cost_grad)
    sorted_ixs = np.argsort(similarity)
    if intensify:
        sorted_ixs = np.flip(sorted_ixs)
    for i in sorted_ixs:
        move = np.copy(idx)
        move[i] = not move[i]
        if index_to_str(move) not in expanded_set:
            return move
    return None


def generate_counts_from_ds(df, group_field):
    x1 = df.groupby([group_field])['item'].count().to_frame()
    x2 = df.groupby([group_field, 'label'])['item'].count().unstack().fillna(0)
    
    x = pd.merge(left=x1, right=x2, left_index=True, right_index=True)
    
    return x.to_numpy(), x


class GradientSolver():

    def __init__(self,
                 problem,
                 candidate,
                 test_size,
                 use_norm_cost=False,
                 min_cost=None,
                 max_empty_iterations=100,
                 max_intensity_iterations=10):
        self.problem = problem
        self.p = test_size
        self.max_empty_iterations = max_empty_iterations
        self.max_intensity_iterations = max_intensity_iterations
        self.min_cost = min_cost or (.0000001 if use_norm_cost else 10000)
        self.incumbent = Solution(calculate_cost(problem, candidate, test_size), candidate)

    def solve(self, verbose=True):
        """
        Uses the gradient solver to calculate the best split.
        @param min_cost: Minimum cost criterion.
        @param max_empty_iterations: Maximum number of non-improving iterations.
        @param max_intensity_iterations: Maximum number of intensity iterations.
        @param verbose: Verbose flag.
        @return: The incumbent solution.
        """
        terminated = False
        intensify = True
        expanded_set = set()
        solution = self.incumbent
        n = 0
        n_intensity = 0

        while not terminated:
            move = generate_cosine_move(
                self.problem, solution.index, self.p, expanded_set, intensify)
            intensify = True
            if move is not None:
                solution = Solution(calculate_cost(self.problem, move, self.p), move)
                expanded_set.add(index_to_str(solution.index))
                if solution.cost < self.incumbent.cost:
                    self.incumbent = solution
                    n = 0
                    n_intensity = 0

                    if verbose:
                        print(self.incumbent.cost)
                else:
                    if n_intensity > self.max_intensity_iterations:
                        intensify = False
                        n_intensity = 0
            else:
                terminated = True
            n += 1
            n_intensity += 1
            if n > self.max_empty_iterations or self.incumbent.cost < self.min_cost:
                terminated = True

        return self.incumbent


def gradient_group_stratify(df, test_size, group_field, random_state=None):
    sample_cnt, sample_cnt_df = generate_counts_from_ds(df, group_field)
    solution_arr = create_initial_solution(sample_cnt, test_size, random_state=random_state)
    g_solver = GradientSolver(sample_cnt, solution_arr, test_size)
    solution = g_solver.solve(verbose=False)

    sample_cnt_df["new_partition"] = pd.Series(
        ['test' if idx == True else 'train' for idx in solution.index], index=sample_cnt_df.index)

    train_locs = sample_cnt_df[sample_cnt_df["new_partition"] == 'train'].index.values
    test_locs = sample_cnt_df[sample_cnt_df["new_partition"] == 'test'].index.values
    train_ords = df[df[group_field].isin(train_locs)].index.values
    test_ords = df[df[group_field].isin(test_locs)].index.values
    return train_ords, test_ords