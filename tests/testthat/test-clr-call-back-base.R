context("test-cyclical-learning-rate-callback.R")

library(keras)
tensorflow::use_session_with_seed(4)
dataset <- dataset_boston_housing()
c(c(train_data, train_targets), c(test_data, test_targets)) %<-% dataset

mean <- apply(train_data, 2, mean)
std <- apply(train_data, 2, sd)
train_data <- scale(train_data, center = mean, scale = std)
test_data <- scale(test_data, center = mean, scale = std)

generate_model <- function() {
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
}



test_that("triangle", {
  callback_clr <- new_callback_cyclical_learning_rate(
    step_size = 6,
    base_lr = 0.001,
    max_lr = 0.006,
    verbose = 0
  )
  generate_model() %>% fit(
    train_data, train_targets,
    validation_data = list(test_data, test_targets),
    epochs = 10, verbose = 0,
    callbacks = list(callback_clr)
  )
  # plot_clr_history(callback_clr)
  expect_known_value(
    callback_clr$history,
    test_path("reference-objects/base-log-triangle")
  )
})

test_that("triangle2", {
  callback_clr <- new_callback_cyclical_learning_rate(
    step_size = 6,
    base_lr = 0.001,
    max_lr = 0.006,
    mode = "triangular2",
    verbose = 0
  )
  generate_model() %>% fit(
    train_data, train_targets,
    validation_data = list(test_data, test_targets),
    epochs = 10, verbose = 0,
    callbacks = list(callback_clr)
  )
  # plot_clr_history(callback_clr)
  expect_known_value(
    callback_clr$history,
    test_path("reference-objects/base-log-triangle2")
  )
})

test_that("exp_range", {
  callback_clr <- new_callback_cyclical_learning_rate(
    step_size = 7,
    base_lr = 0.001,
    max_lr = 0.006,
    gamma = 0.99,
    mode = "exp_range",
    verbose = 0
  )
  generate_model() %>% fit(
    train_data, train_targets,
    validation_data = list(test_data, test_targets),
    epochs = 10, verbose = 0,
    callbacks = list(callback_clr)
  )
  plot_clr_history(callback_clr)
  expect_known_value(
    callback_clr$history,
    test_path("reference-objects/base-log-exp_range")
  )
})


test_that("assertions", {
  expect_error(
    new_callback_cyclical_learning_rate(
      step_size = -1,
      base_lr = 0.001,
      max_lr = 0.006,
      gamma = 0.99,
      mode = "exp_range",
      verbose = 0
    ), ">= 1"
  )

  expect_error(new_callback_cyclical_learning_rate(
    step_size = 6,
    base_lr = 0.01,
    max_lr = 0.006,
    gamma = 0.99,
    mode = "exp_range",
    verbose = 0
  ), ">= 0")

  expect_error(new_callback_cyclical_learning_rate(
    step_size = 6,
    base_lr = 0.001,
    max_lr = 0.006,
    gamma = 0.99,
    mode = "abc",
    verbose = 0
  ), "exp_range")
})


test_that("proper handling when validation data is missing", {
  callback_clr <- new_callback_cyclical_learning_rate(
    step_size = 6,
    base_lr = 0.001,
    max_lr = 0.006,
    verbose = 0
  )
  expect_error(
    fit(generate_model(),
      train_data, train_targets,
      epochs = 2, verbose = 0,
      callbacks = list(callback_clr)
    ), NA
  )


  callback_clr <- new_callback_cyclical_learning_rate(
    step_size = 6,
    base_lr = 0.001,
    max_lr = 0.006,
    verbose = 0, patience = 4
  )
  expect_error(
    fit(generate_model(),
      train_data, train_targets,
      epochs = 2, verbose = 0,
      callbacks = list(callback_clr)
    ), "`validation_data`"
  )
})
