class LGBMEnsembleModel:
    def __init__(self, n_fold=3):
        self.n_fold = n_fold

        def make(i):
            ml = LGBMRegressor(num_leaves=128,
                               max_depth=7,
                               n_estimators=1000,
                               bagging_seed=11*(i+1))
            return ml

        #self.meta_model=LGBMRegressor(num_leaves=128,
        #                              max_depth=9,
        #                              n_estimators=1000,
        #                              objective="regression",
        #                              metric="mean_squared_error")

        self.model_list=[]

    def fit(self, x, y):
        width = len(x) // self.n_fold

        for fold_i in range(self.n_fold):
            start = width * fold_i
            end = start + width

            if type(x) is pd.DataFrame:
                x = x.reset_index(drop=True)
                y = y.reset_index(drop=True)
                valid_x = x.iloc[start:end]
                valid_y = y.iloc[start:end]
                train_x = x.drop(range(len(x))[start:end],axis=0)
                train_y = y.drop(range(len(x))[start:end],axis=0)
            else:
                valid_x = x[start:end,:]
                valid_y = y[start:end,:]
                train_x = np.delete(x,range(len(x))[start:end+1],axis=0)
                train_y = np.delete(y,range(len(x))[start:end+1],axis=0)

            model = LGBMRegressor(num_leaves=128,
                                  max_depth=9,
                                  n_estimators=1000,
                                  objective="regression",
                                  metric="mean_squared_error")

            model.fit(train_x, train_y, eval_set=(valid_x,valid_y), eval_metric="mean_squared_error", early_stopping_rounds=3, verbose=-1)
            self.model_list.append(model)

            if fold_i == 0:
                oof_x = list(model.predict(valid_x, num_iteration = model.best_iteration_).reshape(-1, 1))
                oof_y = list(valid_y)
            else:
                oof_x += list(model.predict(valid_x, num_iteration = model.best_iteration_).reshape(-1, 1))
                oof_y += list(valid_y)

        #self.meta_model.fit(oof_x, oof_y)
        return self

    def predict(self, x):
        y = np.zeros([1000, self.n_fold])
        for ml_i, model in enumerate(self.model_list):
            y[:,ml_i] = model.predict(x, num_iteration = model.best_iteration_)

        y = np.mean(y,axis=1)

        return y
