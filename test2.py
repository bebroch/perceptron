import numpy as np

# W = np.array([[0.1], [0.2], [0.3]])
# b = np.array([[0.1], [0.2]])
# x = np.array([1, 2, 3])
# z = (W @ x) + b.T
# print(z.T)

data = [{"question": [1, 2, 3], "answer": 4}]

for question, answer in data:
    print(question, answer)
