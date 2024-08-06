# Mateus Mesquita de Padua 11921ECP011
import numpy as np
import matplotlib.pyplot as plt

# Construindo a base de observações
np.random.seed(0)
x = np.random.rand(100, 1)
y = 2 + 3 * x + np.random.rand(100, 1)

weights = np.random.rand(2, 1)

learning_rate = 0.01

num_iterations = 1000

costs = []

# Treinando a ADALINE com a base de observações
for i in range(num_iterations):
    x_bias = np.c_[np.ones((x.shape[0], 1)), x]
    y_predict = x_bias.dot(weights)
    error = y_predict - y
    
    cost = (error ** 2).sum() / (2 * x.shape[0])
    costs.append(cost)
    
    weights = weights - learning_rate * x_bias.T.dot(error) / x.shape[0]

y_pred = x_bias.dot(weights)

plt.scatter(x, y, color='b')
plt.plot(x, y_pred, color='k')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

print('Equation of line obtained from ADALINE: y = ', weights[0][0], '+', weights[1][0], 'x')

corr = np.corrcoef(x.T, y.T)[0, 1]
print('Pearson Correlation Coefficient:', corr)

y_mean = np.mean(y)
SST = ((y - y_mean)**2).sum()
SSR = ((y_pred - y_mean)**2).sum()
R_2 = SSR/SST
print('Coefficient of Determination (R^2):', R_2)
