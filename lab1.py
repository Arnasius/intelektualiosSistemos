import numpy as np

# Example data
x1 = np.array([0.21835, 0.14115, 0.37022, 0.14913, 0.18474])
x2 = np.array([0.81884, 0.83535, 0.8111, 0.77104, 0.6279])
P = np.vstack((x1, x2))

T = np.array([1, 1, 1, -1, -1])

# Initialize weights and bias
w1 = np.random.randn()
w2 = np.random.randn()
b = np.random.randn()

# Training parameters
n = 0.1
num_examples = P.shape[1]
e = np.inf
iterations = 0

# Training loop
while e != 0:
    e = 0
    iterations += 1
    for i in range(num_examples):
        xi1 = P[0, i]
        xi2 = P[1, i]
        
        vi = xi1 * w1 + xi2 * w2 + b
        
        y = 1 if vi > 0 else -1
        
        ei = T[i] - y
        
        w1 = w1 + n * ei * xi1
        w2 = w2 + n * ei * xi2
        b = b + n * ei
        
        e = e + abs(ei)

# Display final parameters
print(f'Final w1: {w1}')
print(f'Final w2: {w2}')
print(f'Final b: {b}')
print(f'Iterations: {iterations}')

# Testing part
x1_test = np.array([0.31565, 0.36484, 0.46111, 0.55223, 0.16975, 0.49187, 0.08838, 0.098166])
x2_test = np.array([0.83101, 0.8518, 0.82518, 0.83449, 0.84049, 0.80889, 0.62068, 0.79092])
P_test = np.vstack((x1_test, x2_test))

T_test = np.array([1, 1, 1, 1, 1, 1, -1, -1])

num_test_examples = len(T_test)
correctly_classified = np.zeros(num_test_examples)

for i in range(num_test_examples):
    xi1 = P_test[0, i]
    xi2 = P_test[1, i]
    
    vi = xi1 * w1 + xi2 * w2 + b
    
    y = 1 if vi > 0 else -1
    
    correctly_classified[i] = (y == T_test[i])

accuracy = np.sum(correctly_classified) / num_test_examples * 100
print(f'Test Accuracy: {accuracy:.2f}%')
