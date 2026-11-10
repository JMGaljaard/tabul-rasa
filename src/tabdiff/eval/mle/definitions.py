from xgboost import XGBClassifier, XGBRegressor

_MODELS = {
    "binclass": [  # 184
        #  {
        #      'class': DecisionTreeClassifier, # 48
        #      'kwargs': {
        #          'max_depth': [4, 8, 16, 32],
        #          'min_samples_split': [2, 4, 8],
        #          'min_samples_leaf': [1, 2, 4, 8]
        #      }
        #  },
        #  {
        #      'class': AdaBoostClassifier, # 4
        #      'kwargs': {
        #          'n_estimators': [10, 50, 100, 200]
        #      }
        #  },
        #  {
        #     'class': LogisticRegression, # 36
        #     'kwargs': {
        #          'solver': ['lbfgs'],
        #          'n_jobs': [-1],
        #          'max_iter': [10, 50, 100, 200],
        #          'C': [0.01, 0.1, 1.0],
        #          'tol': [1e-01, 1e-02, 1e-04]
        #      }
        #  },
        # {
        #     'class': MLPClassifier, # 12
        #     'kwargs': {
        #         'hidden_layer_sizes': [(100, ), (200, ), (100, 100)],
        #         'max_iter': [50, 100],
        #         'alpha': [0.0001, 0.001]
        #     }
        # },
        # {
        #     'class': RandomForestClassifier, # 48
        #     'kwargs': {
        #          'max_depth': [8, 16, None],
        #          'min_samples_split': [2, 4, 8],
        #          'min_samples_leaf': [1, 2, 4, 8],
        #         'n_jobs': [-1]
        #     }
        # },
        {
            "class": XGBClassifier,  # 36
            "kwargs": {
                "n_estimators": [10, 50, 100],
                "min_child_weight": [1, 10],
                "max_depth": [5, 10, 20],
                "gamma": [0.0, 1.0],
                "objective": ["binary:logistic"],
                "nthread": [5],
                "tree_method": ["hist"],
                "device": ["cpu"],
            },
        }
    ],
    "multiclass": [  # 132
        # {
        #     'class': MLPClassifier, # 12
        #     'kwargs': {
        #         'hidden_layer_sizes': [(100, ), (200, ), (100, 100)],
        #         'max_iter': [50, 100],
        #         'alpha': [0.0001, 0.001]
        #     }
        # },
        #  {
        #      'class': DecisionTreeClassifier, # 48
        #      'kwargs': {
        #          'max_depth': [4, 8, 16, 32],
        #          'min_samples_split': [2, 4, 8],
        #          'min_samples_leaf': [1, 2, 4, 8]
        #      }
        #  },
        # {
        #     'class': RandomForestClassifier, # 36
        #     'kwargs': {
        #          'max_depth': [8, 16, None],
        #          'min_samples_split': [2, 4, 8],
        #          'min_samples_leaf': [1, 2, 4, 8],
        #          'n_jobs': [-1]
        #     }
        # },
        {
            "class": XGBClassifier,  # 36
            "kwargs": {
                "n_estimators": [10, 50, 100],
                "min_child_weight": [1, 10],
                "max_depth": [5, 10, 20],
                "gamma": [0.0, 1.0],
                "objective": ["binary:logistic"],
                "nthread": [5],
                "tree_method": ["hist"],
                "device": ["cpu"],
            },
        }
    ],
    "regression": [  # 84
        # {
        #     'class': LinearRegression,
        # },
        # {
        #    'class': MLPRegressor, # 12
        #    'kwargs': {
        #        'hidden_layer_sizes': [(100, ), (200, ), (100, 100)],
        #        'max_iter': [50, 100],
        #        'alpha': [0.0001, 0.001]
        #    }
        # },
        {
            "class": XGBRegressor,  # 36
            "kwargs": {
                "n_estimators": [10, 50, 100],
                "min_child_weight": [1, 10],
                "max_depth": [5, 10, 20],
                "gamma": [0.0, 1.0],
                "objective": ["reg:linear"],
                "nthread": [5],
                "tree_method": ["hist"],
                "device": ["cpu"],
            },
        },
        # {
        #     'class': RandomForestRegressor, # 36
        #     'kwargs': {
        #          'max_depth': [8, 16, None],
        #          'min_samples_split': [2, 4, 8],
        #          'min_samples_leaf': [1, 2, 4, 8],
        #          'n_jobs': [-1]
        #     }
        # }
    ],
}
