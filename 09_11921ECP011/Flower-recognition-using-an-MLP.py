# Mateus Mesquita de Pádua 11921ECP011
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
filename = "iris.data"
urllib.request.urlretrieve(url, filename)
iris_data = pd.read_csv(filename, header=None)
iris_data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

X = iris_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = iris_data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

encoder = LabelEncoder()
encoder.fit(y_train)
encoded_y_train = encoder.transform(y_train)

y_train_categorical = to_categorical(encoded_y_train)

model = Sequential()
model.add(Dense(10, input_dim=4, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train_categorical, epochs=100, batch_size=10)

y_pred = model.predict(X_test)

y_pred = np.argmax(y_pred, axis=1)

y_pred = encoder.inverse_transform(y_pred)

accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %.2f' % accuracy)

iris = sns.load_dataset("iris")
sns.scatterplot(x="sepal_length", y="sepal_width", hue="species", data=iris)
plt.show()

new_data = [[5.1, 3.5, 1.4, 0.2]]
predictions = model.predict(new_data)
predicted_class = encoder.inverse_transform(predictions.argmax(axis=1))
print("Predicted class for new data:", predicted_class)
