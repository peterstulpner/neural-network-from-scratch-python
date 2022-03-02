# Neuron takes input from 4 previous nodes in the hidden layer, this adds a unique weight but no extra bias
# This is because each Neuron only has its on specific bias, dependent on the neuron
inputs = [1, 2, 3, 2.5]
weights = [0.2, 0.8, -0.5, 1]
bias = 2

output = bias
for i in range(len(inputs)):
    output += inputs[i] * weights[i]

print("Output: " + str(output))
