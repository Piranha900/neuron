from random import random
import numpy as np

sample_data = [[0,0], [0,1], [1,0], [1,1]]

expected_results = [0, 1, 1, 1]

activation_threshold = .5


weights = np.random.random(2)/1000
bias_weight = np.random.random()/1000

for iteration_num in range(5):
    correct_answers = 0
    for idx, sample in enumerate(sample_data):

        input_vect = np.array(sample)        
        activation_level = np.dot(input_vect, weights) + (bias_weight * 1)
        
        if activation_level > activation_threshold:
            neuron_output = 1
        else:
            neuron_output = 0

        if neuron_output == expected_results[idx]:
            correct_answers += 1
        
        new_weigths = []

        for i, x in enumerate(sample):                   
            new_weigths.append(weights[i] + (expected_results[idx] - neuron_output)* x)

        bias_weight = bias_weight + ((expected_results[idx] - neuron_output) * 1)

        weights = np.array(new_weigths)
    
    print('{} correct answers out of 4, for iteration {}'.format(correct_answers, iteration_num))
