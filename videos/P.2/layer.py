inputs = [1, 2, 3, 2.5]
weights = [[0.2, 0.8, -0.5, 1], [0.5, -0.91,
                                 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]]
bias = [2, 3, 0.5]

output = bias

for j in range(len(weights)):
    for i in range(len(inputs)):
        output[j] += inputs[i] * weights[j][i]

print("Output: " + str(output))
