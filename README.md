
<!-- README.md is generated from README.Rmd. Please edit that file -->

[![lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://www.tidyverse.org/lifecycle/#experimental)
[![Travis build
status](https://travis-ci.org/lorenzwalthert/KerasMisc.svg?branch=master)](https://travis-ci.org/lorenzwalthert/KerasMisc)
[![Coverage
status](https://codecov.io/gh/lorenzwalthert/KerasMisc/branch/master/graph/badge.svg)](https://codecov.io/github/lorenzwalthert/KerasMisc?branch=master)

# KerasMisc

The goal of KerasMisc is to provide a collection of tools that enhance
Keras. Currently, the package features:

  - a Keras callback for cyclical learning rate scheduling as proposed
    by [Smith (2017)](https://arxiv.org/abs/1506.01186), closely adapted
    from the [Python implementation](https://github.com/bckenstler/CLR).
    Please see the [README](https://github.com/bckenstler/CLR) from the
    Python implementation for details. It’s really almost a 1:1
    translation.

## Installation

You can install the development version of KerasMisc from GitHub with

``` r
remotes::install_github("lorenzwalthert/KerasMisc")
```

## Features

**Keras callbacks**

Let’s create a model

``` r
library(keras)
library(KerasMisc)
dataset <- dataset_boston_housing()
c(c(train_data, train_targets), c(test_data, test_targets)) %<-% dataset

mean <- apply(train_data, 2, mean)
std <- apply(train_data, 2, sd)
train_data <- scale(train_data, center = mean, scale = std)
test_data <- scale(test_data, center = mean, scale = std)


model <- keras_model_sequential() %>%
  layer_dense(
    units = 64, activation = "relu",
    input_shape = dim(train_data)[[2]]
  ) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 1)
model %>% compile(
  optimizer = optimizer_rmsprop(lr = 0.001),
  loss = "mse",
  metrics = c("mae")
)
```

Next, we can fit the model with a learning rate schedule.

``` r
callback_clr <- new_callback_cyclical_learning_rate(
  step_size = 32,
  base_lr = 0.001,
  max_lr = 0.006,
  gamma = 0.99,
  mode = "triangular"
)
model %>% fit(
  train_data, train_targets,
  validation_data = list(test_data, test_targets),
  epochs = 10, verbose = 1,
  callbacks = list(callback_clr)
)
```

We can now have a look at the learning rates:

``` r
head(callback_clr$history)
#>           lr iterations epochs
#> 1 0.00100000          0      1
#> 2 0.00115625          1      1
#> 3 0.00131250          2      1
#> 4 0.00146875          3      1
#> 5 0.00162500          4      1
#> 6 0.00178125          5      1
```

The colors in the plot mark the different epochs whereas:

``` r
plot_clr_history(callback_clr)
```

<img src="man/figures/README-plot-clr-1.png" width="100%" />
