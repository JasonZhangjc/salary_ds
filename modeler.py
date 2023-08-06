import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import sklearn



def data_split():
    '''
    split data into training and test sets with features and labels
    '''
    df = pd.read_csv('data_eda.csv')

    # feature selection
    df_model = df[['avg_salary', 'Rating','Size', 'Type of ownership', 'Industry',
                   'Sector', 'Revenue', 'num_comp', 'hourly', 'employer_provided',
                   'us_state', 'same_state', 'age', 'python', 'spark', 'aws', 'excel',
                   'job_simplified', 'seniority', 'jd_len']]


    # get dummy data 
    df_dum = pd.get_dummies(df_model)
    print(df_dum.head())

    # train test split 
    from sklearn.model_selection import train_test_split
    X = df_dum.drop('avg_salary', axis =1)
    y = df_dum['avg_salary'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X, X_train, X_test, y, y_train, y_test



def linear_regression(X_train, y_train):
    '''
    multiple linear regression 
    '''
    # import statsmodels.api as sm
    # X_sm = X = sm.add_constant(X)
    # model = sm.OLS(y, X_sm)
    # model.fit().summary()

    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score

    lm = LinearRegression()
    lm.fit(X_train, y_train)

    print(np.mean(cross_val_score(lm, X_train, y_train, 
                                  scoring = 'neg_mean_absolute_error', cv= 3)))
    
    return lm
    


def lasso_regression(X_train, y_train):
    '''
    lasso regression 
    '''
    from sklearn.linear_model import Lasso
    from sklearn.model_selection import cross_val_score

    lm = Lasso(alpha=.13)
    lm.fit(X_train, y_train)
    np.mean(cross_val_score(lm, X_train, y_train, scoring = 'neg_mean_absolute_error', cv= 3))

    alpha = []
    error = []

    for i in range(1,100):
        alpha.append(i/100)
        lm_ = Lasso(alpha=(i/100))
        error.append(np.mean(cross_val_score(lm_, X_train, y_train, 
                                             scoring = 'neg_mean_absolute_error', cv= 3)))
        
    plt.plot(alpha, error)
    plt.savefig("lasso_regressor_training_process.png")

    err = tuple(zip(alpha, error))
    df_err = pd.DataFrame(err, columns = ['alpha','error'])
    df_err[df_err['error'] == max(df_err['error'])]

    return lm



def random_forest(X_train, y_train):
    '''
    random forest (RF)
    grid search to tune the RF
    '''
    # random forest 
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score

    rf = RandomForestRegressor()

    np.mean(cross_val_score(rf, X_train, y_train, 
                            scoring = 'neg_mean_absolute_error', cv= 3))

    # tune models GridsearchCV 
    from sklearn.model_selection import GridSearchCV
    parameters = {'n_estimators':range(100,200,10), 'criterion':('squared_error','absolute_error'), 
                  'max_features':('sqrt','log2')}
    gs = GridSearchCV(rf, parameters, scoring='neg_mean_absolute_error', cv=3)
    gs.fit(X_train, y_train)

    print(gs.best_score_)
    print(gs.best_estimator_)

    return rf, gs



def test(X_test, y_test, lr1, lr2, gs):
    '''
    test all lr1, lr2, and (rf, gs)
    store the best model
    '''
    # test ensembles 
    tpred_lr1 = lr1.predict(X_test)
    tpred_lr2 = lr2.predict(X_test)
    tpred_rf = gs.best_estimator_.predict(X_test)

    from sklearn.metrics import mean_absolute_error

    print('mean absolute error of lr1 prediction: ', mean_absolute_error(y_test,tpred_lr1))
    print('mean absolute error of lr2 prediction: ', mean_absolute_error(y_test,tpred_lr2))
    print('mean absolute error of rf prediction: ', mean_absolute_error(y_test,tpred_rf))
    print('mean absolute error of (lr1+rf)/2 prediction: ', mean_absolute_error(y_test,(tpred_lr1+tpred_rf)/2))

    # store the best model
    import pickle

    pickl = {'model': gs.best_estimator_}
    pickle.dump( pickl, open( 'model_file' + ".p", "wb" ) )

    

def predict_trial(X_test, y_test):
    '''
    predict with stored model
    '''
    import pickle
    file_name = "model_file.p"
    with open(file_name, 'rb') as pickled:
        data = pickle.load(pickled)
        model = data['model']
    
    # print(list(X_test.iloc[1,:]))
    print(model.predict(np.array(list(X_test.iloc[1,:])).reshape(1,-1))[0])
    