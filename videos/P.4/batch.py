import numpy as np

# Single layer of inputs simply describes inputs at one point in time,
# batching allows to show mulitple at a time, which allows for more generalisation
# The bigger the batch, the less loss you have and the better generalisation is achieved, to a point (32-64 is common)

batch = [[1, 2, 3, 2.5], [2, 5, -1, 2], [-1.5, 2.7, 3.3, -0.8]]

# Layer: Nothin needs to be changed as the batch is independent of the netowrk layer
weights = [[0.2, 0.8, -0.5, 1], [0.5, -0.91,
                                 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]


# Weights: 3 x 4
# Batch:   3 x 4
# Hence dimensions don't match, dot product needs to be changed, this is solved through transposing weights matrix

output = np.dot(batch, np.array(weights).T) + biases

print(f"Output: {output}")
