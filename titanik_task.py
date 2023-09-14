import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from keras import Sequential
from keras import layers

path = '/home/vadim/Desktop/titanic_task/'

def prepare_data(path):
    

    test_data, test_labels = pd.read_csv(path+'test.csv'), np.asarray(pd.read_csv(path+'gender_submission.csv')).astype('float32')
    test_data_index = test_data['PassengerId'].values
    test_data = test_data.drop(columns=['PassengerId', 'Name', 'Ticket'], axis=1)
    train_data = pd.read_csv(path+'train.csv')
    train_labels = np.asarray(train_data['Survived']).astype('float32')
    train_data = train_data.drop(columns=['PassengerId', 'Survived', 'Name', 'Cabin' , 'Ticket'], axis=1)
    
    train_data[['Sex']] = train_data[['Sex']].apply(lambda x: pd.factorize(x)[0])
    train_data[['Fare']] = train_data[['Fare']].fillna(train_data['Fare'].mean()).apply(lambda y: round(y/1000, 4))
    train_data[['Age']] = train_data[['Age']].fillna(round(train_data['Age'].mean())).apply(lambda z: z/100)

    test_data[['Sex']] = test_data[['Sex']].apply(lambda x: pd.factorize(x)[0])
    test_data[['Fare']] = test_data[['Fare']].fillna(test_data['Fare'].mean()).apply(lambda y: round(y/1000, 4))
    test_data[['Age']] = test_data[['Age']].fillna(round(test_data['Age'].mean())).apply(lambda z: z/100)

    test_data.dropna(subset=['Embarked'])

    onehotenc,onehotenc1,onehotenc2 = OneHotEncoder(),OneHotEncoder(),OneHotEncoder()
    onehotenc_test,onehotenc_test1,onehotenc_test2 = OneHotEncoder(),OneHotEncoder(),OneHotEncoder()

    data_new,data_new1,data_new2 = onehotenc.fit_transform(train_data[['Embarked']].values),onehotenc1.fit_transform(train_data[['Parch']].astype('category').values),onehotenc2.fit_transform(train_data[['Pclass']].astype('category').values)
    data_new_test,data_new_test1,data_new_test2 = onehotenc_test.fit_transform(test_data[['Embarked']].values),onehotenc_test1.fit_transform(test_data[['Parch']].astype('category').values),onehotenc_test2.fit_transform(test_data[['Pclass']].astype('category').values)

    data_new,data_new1,data_new2 = pd.DataFrame(data_new.toarray(),columns=onehotenc.categories_),pd.DataFrame(data_new1.toarray(),columns=onehotenc1.categories_),pd.DataFrame(data_new2.toarray(),columns=onehotenc2.categories_)
    data_new_test,data_new_test1,data_new_test2 = pd.DataFrame(data_new_test.toarray(),columns=onehotenc_test.categories_),pd.DataFrame(data_new_test1.toarray(),columns=onehotenc_test1.categories_),pd.DataFrame(data_new_test2.toarray(),columns=onehotenc_test2.categories_)

    train_data = pd.concat([train_data, data_new, data_new1, data_new2], sort=False, axis=1).drop(columns = ['Embarked','Parch', 'Pclass'])
    train_data = train_data.to_numpy()

    test_data = pd.concat([test_data, data_new_test, data_new_test1, data_new_test2], sort=False, axis=1).drop(columns = ['Embarked','Parch', 'Pclass','Cabin'])
    test_data = np.asarray(test_data.to_numpy()).astype('float32')

    return [train_data, train_labels, test_data, test_labels, test_data_index]






model = Sequential([
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid'),
])

model.compile(optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'])

model.fit(prepare_data(path= path)[0],
    prepare_data(path= path)[1],
    epochs=100,
    batch_size=512,
    validation_split=0.4)


results = model.predict(prepare_data(path=path)[2],batch_size=128)
results = [1 if i>=0.5 else 0 for i in results]
output = pd.DataFrame({'PassengerId': prepare_data(path=path)[4], 'Survived': results})
output.to_csv('titanic_solution.csv', index=False)

# import matplotlib.pyplot as plt
# history_dict = history.history
# acc_values = history_dict["accuracy"]

# val_acc_values = history_dict["val_accuracy"]

# epochs = range(1,101)
# plt.plot(epochs, acc_values, "bo", label="Точность на этапе обучения")
# plt.plot(epochs, val_acc_values, "b", label="Точность на этапе проверки")
# plt.title("Точночть на этапах обучения и проверки")

# plt.xlabel("Эпохи")

# plt.ylabel("Точность")
# plt.legend()
# plt.show()
