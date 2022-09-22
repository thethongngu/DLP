from dataloader import getData

import numpy as np

inputs, labels = getData('train')

label_count = [0, 0, 0, 0, 0]
for label in labels:
    label_count[label] += 1

print(label_count)

inputs, labels = getData('test')

label_count1 = [0, 0, 0, 0, 0]
for label in labels:
    label_count1[label] += 1

print(label_count1)
print(np.array(label_count) / np.array(label_count1))

for val in label_count:
    print(val / np.sum(np.array(label_count)) * 100)
