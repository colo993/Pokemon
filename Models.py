import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression


# Loading data set
pokemon = pd.read_csv('D:/Python/My projects - Data Science/Pokemon/Pokemon_for_ML.csv') 

# First I will drop rows with NA values, they can cause an errors in models
pokemon.dropna(axis=0, how='any')

# We don't need to first columns, whith below code I erase them from dataset
pokemon.drop(pokemon.columns[[0, 1]], axis=1, inplace=True)

##################################### Linear Regression #########################################
# Here is an example how to easy implement Linear Regression model
# I am using data sets of 'Attack' and 'HP' values

# assign values to X, y and reshaping X value
X = pokemon['Attack'].values.reshape(-1, 1)
y = pokemon['HP'].values

# plotting scatter plot of values X, y
plt.scatter(X, y, s=3)
plt.xlabel('Attack')
plt.ylabel('HP')

# assign linear regression to the value
lin_reg = LinearRegression()

# fit the model 
lin_reg.fit(X.reshape(-1, 1),y)

# Here you obtain the coefficient of determination (𝑅²), which means accuracy of your model 
# print('Linear Regression train accuracy:', lin_reg.score(X_train, y_train))

# The linear regression has very poor result, but you can see how very easy you can use this model.

# .intercept_, which represents the coefficient, 𝑏₀ and .coef_, which represents 𝑏₁:
print ('intercept: ', lin_reg.intercept_) 
print('coefficient:', lin_reg.coef_)      
# The value 𝑏₀ = 43.41 (approximately) illustrates that your model predicts the response 43.41 when 𝑥 is zero.
# The value 𝑏₁ = 0.32 means that the predicted response rises by 0.32 when 𝑥 is increased by one.


# predictions of y values('HP') on feature values X (Attack)
y_pred = lin_reg.predict(X.reshape(-1, 1))   

# plotting raw data with fitted mode predictions on data:
plt.figure()
plt.scatter(X, y, s=3)
plt.plot(X, y_pred, 'r')
plt.xlabel('Attack')
plt.ylabel('HP')

# This model maybe it is not the best option for this purpose but you can see, that implementation of
# Linear regression is not so hard, and expression is easy to read and understand.

####################################################################
# Let's try now use linear regression model with different data

# assign values to X, y and reshaping X value
X = pokemon['Speed'].values.reshape(-1, 1)
y = pokemon['Win Percentage'].values

# plotting scatter plot of values X, y
plt.figure()
plt.scatter(X, y, s=3)
plt.xlabel('Speed')
plt.ylabel('Win Percentage')

# assign linear regression to the value
lin_reg = LinearRegression()

# fit the model 
lin_reg.fit(X.reshape(-1, 1),y)

# Here you obtain the coefficient of determination (𝑅²), which means accuracy of your model 
lin_reg_accuracy = lin_reg.score(X, y)
print('Linear Regression accuracy score:', lin_reg_accuracy)
# The linear regression has very poor result, but you can see how very easy you can use this model.

# .intercept_, which represents the coefficient, 𝑏₀ and .coef_, which represents 𝑏₁:
print ('intercept: ', lin_reg.intercept_) 
print('coefficient:', lin_reg.coef_)      
# The value 𝑏₀ = -0.05 (approximately) illustrates that your model predicts the response -0.05 when 𝑥 is zero.
# The value 𝑏₁ = 0.0082 means that the predicted response rises by 0.0082 when 𝑥 is increased by one.


# predictions of y values('Win %') on feature values X (Attack)
y_pred = lin_reg.predict(X.reshape(-1, 1))   

# plotting raw data with fitted mode predictions on data:
plt.figure()
plt.scatter(X, y, s=3)
plt.plot(X, y_pred, 'r')
plt.xlabel('Attack')
plt.ylabel('HP')

# here we can see that Accuracy score in this model is about 0.87, which means model predicts quite well variables 

###################################################################################################################
# Creating correlation table
# First I create Data Frame which I will use in correlation table
df = pd.DataFrame(pokemon, columns=['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Total Fights'])

corrMatrix = df.corr()
plt.figure()
sns.heatmap(corrMatrix, annot=True)
plt.title('Pokemon Feature Correlation')
plt.xticks(rotation=45)
plt.show()



# Splitting the dataset into the Training set and Test set
X = df.iloc[:, 0:6].values
y = df.iloc[:, 6].values

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# check the shape of our training and testing data
print('X_train shape:' + str(X_train.shape))
print('X_test shape:' + str(X_test.shape))
print('y_train shape:' + str(y_train.shape))
print('y_test shape:' + str(y_test.shape))

############################################# Logistic Reggresion #################################

# Import logistic regression
from sklearn.linear_model import LogisticRegression


# Most features below of Logistic regression stay default, but we write them down to record them 
my_lr = LogisticRegression(C=0.1, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, max_iter=100, 
                           multi_class='auto', n_jobs=None, penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
                           verbose=0, warm_start=False)

# Training the model
my_lr.fit(X_train, y_train)

# Using trained model to make predictions on the features of the samples from the held-out test set:
y_pred = my_lr.predict(X_test)


# Checking accuracy of model
print('Accuracy of Logistic regression: ' + str(my_lr.score(X_train, y_train)))

# from sklearn.metrics import classification_report
# print(classification_report(y_test, y_pred))


############################################## Linear Regression ############################################

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression() 

lin_reg.fit(X_train, y_train)

y_pred = lin_reg.predict(X_test)

print('Accuracy of Linear regression: ' + str(lin_reg.score(X_train, y_train)))


#################################################### Decision Tree #################################################

from sklearn.tree import DecisionTreeRegressor

decision_tree = DecisionTreeRegressor(random_state=0, max_depth=28) # I tried different max_depth, and 28 looks like i's the most
                                                                      # efficient, over 28 we dont get significant performence difference
                                                                      # so we don't have to overload our CPU by more computation

decision_tree.fit(X_train, y_train)

print('Accuracy of Decision Tree: ' + str(decision_tree.score(X_train, y_train)))

# Predicting a new results
y_pred = decision_tree.predict(X_test)

# calculating mean absolut error
mean_error = mean_absolute_error(y_test, y_pred)

"""# plotting our decision tree
from sklearn import tree
fig = plt.figure(figsize=(25,20))
tree.plot_tree(decision_tree, feature_names=X, class_names=y, filled=True)"""
# There is too many steps and graph is unreadable


################################################### Random Forest #################################################

from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators=10, random_state=0)

forest_reg.fit(X_train, y_train)

print("Accuracy of Random Forest:" + str(forest_reg.score(X_train, y_train)))

# Using predict function on test data
y_pred = forest_reg.predict(X_test)

# calculating absolute error
errors = abs(y_pred - y_test)

# calculating mean absolut error
mean_error = mean_absolute_error(y_test, y_pred)
round(np.mean(errors), 2), 'degrees.'

mape = 100 * (errors/y_test)
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')









