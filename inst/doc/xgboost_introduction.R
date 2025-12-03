## -----------------------------------------------------------------------------
library(xgboost)
data(ToothGrowth)

y <- ToothGrowth$supp # the response which we want to model/predict
x <- ToothGrowth[, c("len", "dose")] # the features from which we want to predct it
model <- xgboost(x, y, nthreads = 1, nrounds = 2)
model

## -----------------------------------------------------------------------------
predict(model, x[1:6, ], type = "response") # probabilities for y's last level ("VC")
predict(model, x[1:6, ], type = "raw")      # log-odds
predict(model, x[1:6, ], type = "class")    # class with highest probability

## -----------------------------------------------------------------------------
data(mtcars)

y <- mtcars$mpg
x <- mtcars[, -1]
model_gaussian <- xgboost(x, y, nthreads = 1, nrounds = 2) # default is squared loss (Gaussian)
model_poisson <- xgboost(x, y, objective = "count:poisson", nthreads = 1, nrounds = 2)
model_abserr <- xgboost(x, y, objective = "reg:absoluteerror", nthreads = 1, nrounds = 2)

## -----------------------------------------------------------------------------
y <- ToothGrowth$supp
x <- ToothGrowth[, c("len", "dose")]
model_conservative <- xgboost(
    x, y, nthreads = 1,
    nrounds = 5,
    max_depth = 2,
    reg_lambda = 0.5,
    learning_rate = 0.15
)
pred_conservative <- predict(
    model_conservative,
    x
)
pred_conservative[1:6] # probabilities are all closer to 0.5 now

## -----------------------------------------------------------------------------
xgboost(
    x, y, nthreads = 1,
    eval_set = 0.2,
    monitor_training = TRUE,
    verbosity = 1,
    eval_metric = c("auc", "logloss"),
    nrounds = 5,
    max_depth = 2,
    reg_lambda = 0.5,
    learning_rate = 0.15
)

## -----------------------------------------------------------------------------
attributes(model)

## -----------------------------------------------------------------------------
xgb.attributes(model)

## -----------------------------------------------------------------------------
xgb.importance(model)

## -----------------------------------------------------------------------------
xgb.model.dt.tree(model)

## -----------------------------------------------------------------------------
data("agaricus.train")
dmatrix <- xgb.DMatrix(
    data = agaricus.train$data,  # a sparse CSC matrix ('dgCMatrix')
    label = agaricus.train$label, # zeros and ones
    nthread = 1
)
booster <- xgb.train(
    data = dmatrix,
    nrounds = 10,
    params = xgb.params(
        objective = "binary:logistic",
        nthread = 1,
        max_depth = 3
    )
)

data("agaricus.test")
dmatrix_test <- xgb.DMatrix(agaricus.test$data, nthread = 1)
pred_prob <- predict(booster, dmatrix_test)
pred_raw <- predict(booster, dmatrix_test, outputmargin = TRUE)

## -----------------------------------------------------------------------------
xgb.importance(model)
xgb.importance(booster)

