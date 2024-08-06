# Mateus Mesquita de Padua
# 11921ECP011
import numpy as np
import matplotlib.pyplot as plt

data = np.array([
    [2.215, 2.063, -1],
    [0.224, 1.586, 1],
    [0.294, 0.651, 1],
    [2.327, 2.932, -1],
    [2.497, 2.322, -1],
    [0.169, 1.943, 1],
    [1.274, 2.428, -1],
    [1.526, 0.596, 1],
    [2.009, 2.161, -1],
    [1.759, 0.342, 1],
    [1.367, 0.938, 1],
    [2.173, 2.719, -1],
    [0.856, 1.904, 1],
    [2.21, 1.868, -1],
    [1.587, 1.642, -1],
    [0.35, 0.84, 1],
    [1.441, 0.09, 1],
    [0.185, 1.327, 1],
    [2.764, 1.149, -1],
    [1.947, 1.598, -1]
])

X = data[:, :2]
y = data[:, 2]

# coluna de 1 para X para incluir o bias
X_bias = np.c_[np.ones((X.shape[0], 1)), X]

weights = np.random.rand(3, 1)

learning_rate = 0.01

num_iterations = 1000

errors = []

# Treinando a ADALINE
for i in range(num_iterations):
    output = X_bias.dot(weights)
    error = output - y.reshape(-1, 1)
    # Calculando o erro quadrático médio (EQM)
    mse = (error ** 2).mean()
    
    errors.append(mse)
    
    weights -= learning_rate * X_bias.T.dot(error) / X.shape[0]

# Plotando o erro quadrático médio durante o treinamento
plt.plot(errors)
plt.xlabel('Número de iterações')
plt.ylabel('Erro Quadrático Médio')
plt.title('Erro Quadrático Médio durante o Treinamento')
plt.show()
