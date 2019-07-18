class outlier_remover:
    def __init__(self, out_val_list, model, max_depth=8, min_samples_leaf=1, min_samples_split=5, feature_num=8):
        self.out_val_list_ = out_val_list
        self.model_name_ = model
        self.feature_num_ = feature_num
        
        if model=="lgb":
            self.model = LGBMClassifier(max_depth=4,
                                        num_leaves=64,
                                        min_samples_leaf=min_samples_leaf,
                                        min_samples_split=min_samples_split,
                                        bagging_seed=123,
                                        n_estimators=1000,
                                        objective="multiclass",
                                        varbose=30)
        if model=="simple_tree":
            self.model = DecisionTreeClassifier(max_depth=max_depth,
                                                min_samples_leaf=min_samples_leaf,
                                                min_samples_split=min_samples_split)
        
        
    def fit(self,x,y,cv=False,n_fold=3):
        self.mod_y_set = {}
        self.mean_dict = {}
        self.covar_dict = {}
        self.use_feature_list_ = {}
        self.feature_importances_ = []
        
        if cv:
            self.train_score = []
            self.valid_score = []
            
        #y調整
        y = y.astype(int)
        outlier_dict = {val : i+1 for i,val in enumerate(self.out_val_list_)}
        
        mod_y = []
        for y_val in y:
            if y_val in self.out_val_list_:
                mod_y.append(outlier_dict[y_val])
            else:
                mod_y.append(0)
        self.mod_y_set = set(mod_y)
        
        self.model.fit(x, mod_y)
        importance_dict = {col_name : importance for col_name, importance in zip(x.columns, self.model.feature_importances_)}
        sorted_importance_list = sorted(importance_dict.items(), key=lambda x:x[1], reverse=True)
        sorted_importance_list = [tupple_[0] for tupple_ in sorted_importance_list]
        self.use_feature_list_ = sorted_importance_list[:self.feature_num_]
        
        outlier_means_each_feature = {}
        df = x.copy()
        df["target"] = mod_y
        
        group = df.groupby("target")
        group_mean = group[self.use_feature_list_].mean()
        group_mean.reset_index(inplace=True)
        
        for outlier_val in self.mod_y_set:
            for feature_name in self.use_feature_list_:
                col_name = "{}_{}_dist_from_mean".format("outlier"+str(outlier_val), feature_name)
                mean = group_mean[group_mean["target"]==outlier_val][feature_name].values[0]
                self.mean_dict.update({col_name : mean})
                df[col_name] = abs(df[feature_name] - mean)
        
        feature_combi_list = itertools.combinations(self.use_feature_list_,2)
        for outlier_val in self.mod_y_set:
            for feature_combi in feature_combi_list:
                feature_0 = feature_combi[0]
                feature_1 = feature_combi[1]
                dist_mean_name_0 = "{}_{}_dist_from_mean".format("outlier"+str(outlier_val), feature_0)
                dist_mean_name_1 = "{}_{}_dist_from_mean".format("outlier"+str(outlier_val), feature_1)
                covar = sum(df[dist_mean_name_0]*df[dist_mean_name_1])/len(df) + 1e-3
                col_name = "{}_{}_{}_div_covar".format("outlier"+str(outlier_val), feature_0, feature_1)
                self.covar_dict.update({col_name : covar})
                df[col_name] = df[dist_mean_name_0]*df[dist_mean_name_1]/covar + 1e-3
                
        feature_cols = list(df.columns)
        feature_cols.remove("target")
        #tmp_list = []
        #for col_name in feature_cols:
        #    if "mean" in col_name:
        #        pass
        #    else:
        #        tmp_list.append(col_name)
        #feature_cols = tmp_list
        self.feature_cols = feature_cols
        
        fold = StratifiedKFold(n_splits=n_fold, random_state=123, shuffle=True)
        for fold_iter, (train_idx, valid_idx) in enumerate(fold.split(df, df["target"])):
            train_X = df.iloc[train_idx][feature_cols]
            train_y = df.iloc[train_idx]["target"]
            valid_X = df.iloc[valid_idx][feature_cols]
            valid_y = df.iloc[valid_idx]["target"]
            
            if self.model_name_=="lgb":
                self.model.fit(train_X, train_y, eval_set=(valid_X, valid_y), early_stopping_rounds=10)
            if self.model_name_=="simple_tree":
                self.model.fit(train_X, train_y)
            
            if cv:
                if self.model_name_=="lgb":
                    train_pred = self.model.predict(train_X, num_iteration=self.model.best_iteration_)
                    valid_pred = self.model.predict(valid_X, num_iteration=self.model.best_iteration_)
                if self.model_name_=="simple_tree":
                    train_pred = self.model.predict(train_X)
                    valid_pred = self.model.predict(valid_X)
                    
                self.train_score.append(f1_score(train_y, train_pred, average="macro"))
                self.valid_score.append(f1_score(valid_y, valid_pred, average="macro"))
        self.train_score = np.mean(self.train_score)
        self.valid_score = np.mean(self.valid_score)
        
        new_importance_dict = {col_name : importance for col_name, importance in zip(feature_cols, self.model.feature_importances_)}
        sorted_new_importance_list = sorted(new_importance_dict.items(), key=lambda x:x[1], reverse=True)
        self.feature_importances_ = {_[0] : np.round(_[1], decimals=5 ) for _ in sorted_new_importance_list}
        
    def predict(self,x):
        for outlier_val in self.mod_y_set:
            for feature_name in self.use_feature_list_:
                col_name = "{}_{}_dist_from_mean".format("outlier"+str(outlier_val), feature_name)
                x[col_name] = abs(x[feature_name] - self.mean_dict[col_name])
                
        feature_combi_list = itertools.combinations(self.use_feature_list_,2)
        for outlier_val in self.mod_y_set:
            for feature_combi in feature_combi_list:
                feature_0 = feature_combi[0]
                feature_1 = feature_combi[1]
                dist_mean_name_0 = "{}_{}_dist_from_mean".format("outlier"+str(outlier_val), feature_0)
                dist_mean_name_1 = "{}_{}_dist_from_mean".format("outlier"+str(outlier_val), feature_1)
                col_name = "{}_{}_{}_div_covar".format("outlier"+str(outlier_val), feature_0, feature_1)
                x[col_name] = x[dist_mean_name_0]*df[dist_mean_name_1]/self.covar_dict[col_name] + 1e-3
                
        pred = self.model.predict(x[self.feature_cols], num_iteration=self.model.best_iteration_)
        return pred
