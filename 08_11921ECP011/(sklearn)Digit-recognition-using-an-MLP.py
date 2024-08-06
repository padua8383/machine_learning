# Mateus Mesquita 11921ECP011
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

digits = load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)

mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)

accuracy = np.mean(y_pred == y_test)
print("Acurácia: {:.2f}%".format(accuracy * 100))

fig, axes = plt.subplots(4, 4, figsize=(8, 8))
axes = axes.ravel()

for i, (ax, wrong) in enumerate(zip(axes, np.where(y_pred != y_test)[0])):
    ax.imshow(X_test[wrong].reshape(8, 8), cmap=plt.cm.gray_r)
    ax.set_title("Predicted: {}\nTrue: {}".format(y_pred[wrong], y_test[wrong]))

plt.suptitle("Alguns exemplos de dígitos classificados incorretamente")
plt.show()

plt.plot(mlp.loss_curve_)
plt.title("Evolução do erro durante o treinamento")
plt.xlabel("Épocas")
plt.ylabel("Erro")
plt.show()
