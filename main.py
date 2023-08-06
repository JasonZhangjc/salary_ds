# import scraper as gs
import cleaner
import modeler



if __name__ == '__main__':
    # The scaper worked before, but not anymore.
    # I remain it for reference
    # path = "C:/py/ds/sde_salary/chromedriver.exe"
    # sleep_time = 15
    # df = gs.get_jobs('data scientist', 15, False, path, sleep_time)
    
    # data cleaning
    cleaner.data_clean()

    # train test split
    X, X_train, X_test, y, y_train, y_test = modeler.data_split()

    ### Once the model_file.p exists in the folder, 
    ### do not need to run the following codes
    # train different models
    lr1 = modeler.linear_regression(X_train, y_train)
    lr2 = modeler.lasso_regression(X_train, y_train)
    rf, gs = modeler.random_forest(X_train, y_train)

    # test different models and store the best model
    modeler.test(X_test, y_test, lr1, lr2, gs)

    # predict with the best model
    modeler.predict_trial(X_test, y_test)