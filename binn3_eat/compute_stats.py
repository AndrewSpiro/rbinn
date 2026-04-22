import numpy as np

accuracies = [
    [85.5, 62.2], # binn3_eat/Rescifar10_backup/FGSM/results_rgbedge.txt lines 5 &6
    [74.7, 53.5], # binn3_eat/Rescifar10/FGSM/results_rgbedge_backup1.txt lines 5&6
    [74.22, 54.3] # binn3_eat/Rescifar10/FGSM/results_rgbedge.txt lines 13&14
]

means = np.mean(accuracies, axis = 0)
stds = np.std(accuracies, axis = 0)
stats = zip(means, stds)
print(list(stats))