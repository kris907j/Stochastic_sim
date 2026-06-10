import numpy as np
import matplotlib.pyplot as plt

class RandomnessTests:

    @staticmethod
    def LCG(x0, a, c, M, N):
        xmi = x0
        rando, rando_Ui = [], []

        rando.append(xmi)
        rando_Ui.append(xmi / M)

        for _ in range(N):
            xi = (a * xmi + c) % M
            rando.append(xi)
            rando_Ui.append(xi / M)
            xmi = xi

        return rando, rando_Ui

    @staticmethod
    def chi_squared_test(observed, expected):
        T = sum((o - e) ** 2 / e for o, e in zip(observed, expected))
        df = len(observed) - 1
        return T, df

    @staticmethod
    def KS_test(data, case):
        data = np.sort(data)
        n = len(data)

        Dn = 0

        for i, x in enumerate(data, start=1):
            Fn = i / n
            F = x  # Uniform distribution
            Dn = max(Dn, abs(Fn - F))

        if case == 'known':
            adjusted_test_stat = (
                np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)
            ) * Dn
        elif case == 'normal':
            adjusted_test_stat = (
                np.sqrt(n) - 0.01 + 0.85 / np.sqrt(n)
            ) * Dn
        elif case == 'exp':
            adjusted_test_stat = (
                np.sqrt(n) + 0.26 + 0.5 / np.sqrt(n)
            ) * (Dn - 0.2 / n)
        else:
            raise ValueError(
                "case skal være 'known', 'normal' eller 'exp'"
            )

        print(f"Kolmogorov-Smirnov test statistic: {Dn}")
        print(
            f"Adjusted Kolmogorov-Smirnov test statistic: "
            f"{adjusted_test_stat}"
        )

        return Dn, adjusted_test_stat

    @staticmethod
    def run_wald_wolf(data):
        median = np.median(data)

        seq = ['A' if x > median else 'B' for x in data]

        count_below = seq.count('A')
        count_above = seq.count('B')

        T = 1
        for i in range(1, len(seq)):
            if seq[i] != seq[i - 1]:
                T += 1

        expectation = (
            2 * (count_above * count_below)
            / (count_above + count_below)
        )

        variance = (
            2 * count_above * count_below *
            (2 * count_above * count_below -
             count_above - count_below)
        ) / (
            (count_above + count_below) ** 2 *
            (count_above + count_below - 1)
        )

        z = (T - expectation) / np.sqrt(variance)

        print(f"Number of runs: {T}")
        print(f"Z-score: {z}")

        return T, z

    @staticmethod
    def run_knuth(data):
        T = 1
        run_length = []
        start = 1

        for i in range(1, len(data)):
            if data[i] < data[i - 1]:
                T += 1
                run_length.append(start)
                start = 1
            else:
                start += 1

        run_length.append(start)

        R = np.zeros(6)

        for r in run_length:
            if r >= 6:
                R[5] += 1
            else:
                R[r - 1] += 1

        n = len(data)

        B = np.array([
            1 / 6,
            5 / 24,
            11 / 120,
            19 / 720,
            29 / 5040,
            1 / 840
        ])

        A = np.array([
            [4529.4, 9044.9, 13568, 18091, 22615, 27892],
            [9044.9, 18097, 27139, 36187, 45234, 55789],
            [13568, 27139, 40721, 54281, 67852, 83685],
            [18091, 36187, 54281, 72414, 90470, 111580],
            [22615, 45234, 67852, 90470, 113262, 139476],
            [27892, 55789, 83685, 111580, 139476, 172860]
        ])

        D = R - n * B

        Z = (1 / (n - 6)) * (D.T @ A @ D)

        print(f"Number of runs: {T}")
        print(f"Z-score: {Z}")

        return T, Z

    @staticmethod
    def run_up_down(data):
        signs = []

        for i in range(1, len(data)):
            if data[i] < data[i - 1]:
                signs.append('<')
            else:
                signs.append('>')

        run_length = []
        current = 1

        for i in range(1, len(signs)):
            if signs[i] == signs[i - 1]:
                current += 1
            else:
                run_length.append(current)
                current = 1

        run_length.append(current)

        runs = len(run_length)
        n = len(data)

        expected_runs = (2 * n - 1) / 3
        variance_runs = (16 * n - 29) / 90

        Z = (
            runs - expected_runs
        ) / np.sqrt(variance_runs)

        print(f"Number of runs: {runs}")
        print(f"Z-score: {Z}")

        return runs, Z

    @staticmethod
    def correlation_test(data, lag):
        n = len(data)

        total = 0
        for i in range(1, n - lag):
            total += data[i] * data[i + lag]

        Ch_calculated = total / (n - lag)

        expectation = 1 / 4
        variance = 7 / (144 * n)

        z = (
            Ch_calculated - expectation
        ) / np.sqrt(variance)

        return Ch_calculated, z
    
    @staticmethod
    def analyze(
        data,
        bins=10,
        ks_case="known",
        correlation_lags=(1, 2, 5, 10),
        histogram=True,
        scatter=True,
        chi_square=True,
        ks=True,
        wald_wolf=True,
        knuth=True,
        up_down=True,
        correlation=True,
    ):
        if histogram:
            plt.figure()
            plt.hist(data, bins=bins)
            plt.title("Histogram")
            plt.show()

        if scatter:
            plt.figure()
            plt.scatter(data[:-1], data[1:])
            plt.xlabel("Ui")
            plt.ylabel("Ui+1")
            plt.title("Scatterplot")
            plt.show()

        results = {}

        if chi_square:
            observed, _ = np.histogram(data, bins=bins)
            expected = [len(data) / bins] * bins

            T, df = RandomnessTests.chi_squared_test(observed, expected)
            results["chi_square"] = (T, df)

        if ks:
            results["ks"] = RandomnessTests.KS_test(data, ks_case)

        if wald_wolf:
            results["wald_wolf"] = RandomnessTests.run_wald_wolf(data)

        if knuth:
            results["knuth"] = RandomnessTests.run_knuth(data)

        if up_down:
            results["up_down"] = RandomnessTests.run_up_down(data)

        if correlation:
            corr_results = []
            for lag in correlation_lags:
                corr_results.append(
                    RandomnessTests.correlation_test(data, lag)
                )
            results["correlation"] = corr_results

        return results
                
    @staticmethod
    def compare_datasets(datasets, labels=None, **kwargs):
        if labels is None:
            labels = [f"data_{i}" for i in range(len(datasets))]

        all_results = {}

        for label, data in zip(labels, datasets):
            print(f"\n==================== {label} ====================")
            all_results[label] = RandomnessTests.analyze(data, **kwargs)

        return all_results