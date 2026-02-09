import numpy as np

class GetData:
    def __init__(self, n_instance: int, n_items: int, knapsack_capacity: int):
        self.n_instance = n_instance
        self.n_items = n_items
        self.knapsack_capacity = knapsack_capacity

    def generate_instances(self):
        """
        Generates only:
          1) uncorrelated
          2) weakly correlated
          3) strongly correlated
        using the same formulas as Pisinger.
        """
        np.random.seed(2024)
        instance_data = []

        n = self.n_items
        c = self.knapsack_capacity

        # Choose a data range R automatically.
        # (Formulas still exactly match Pisinger; this just picks a sensible R for a fixed-c setup.)
        R = c/2 +10        # keep all weights <= capacity

        delta = R // 10       # integer version of R/10 used in the paper formulas

        for k in range(self.n_instance):
            kind = k % 3  # 0=uncorrelated, 1=weakly, 2=strongly

            # weights: w_j ~ U[1, R]
            w = np.random.randint(1, R + 1, size=n)

            if kind == 0:
                # Uncorrelated: p_j ~ U[1, R]
                p = np.random.randint(1, R + 1, size=n)

            elif kind == 1:
                # Weakly correlated: p_j ~ U[w_j - R/10, w_j + R/10], with p_j >= 1
                p_list = []
                for wj in w.tolist():
                    low = max(1, wj - delta)
                    high = wj + delta
                    p_list.append(int(np.random.randint(low, high + 1)))
                p = np.array(p_list, dtype=int)

            else:
                # Strongly correlated: p_j = w_j + R/10
                p = w + delta

            instance_data.append((w.astype(int).tolist(), p.astype(int).tolist(), c))

        return instance_data
