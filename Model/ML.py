from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import sklearn.metrics as k
import Input


def MLR(x_train,x_test,y_train,y_test):
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(k.root_mean_squared_error(y_test, y_pred))

    return y_pred

#############Implement Accuracy Models##########
def KNN(x_train,x_test,y_train,y_test):
    knn_model = KNeighborsRegressor(n_neighbors=2)
    knn_model.fit(x_train,y_train)
    y_pred=knn_model.predict(x_test)
    print(k.root_mean_squared_error(y_test,y_pred))
    print(k.mean_squared_error(y_test,y_pred))
    return y_pred



x_test_norm,x_train_norm,y_train,y_test= Input.main()
y_pred = MLR(x_train_norm,x_test_norm,y_train,y_test)
t_pred = KNN(x_train_norm,x_test_norm,y_train,y_test)