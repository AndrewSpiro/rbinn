import numpy as np

accuracies = [
    [85.5, 62.2],
    [74.7, 53.5],
    [74.22, 54.3]
]

means = np.mean(accuracies, axis = 0)
stds = np.std(accuracies, axis = 0)
stats = zip(means, stds)
print(list(stats))