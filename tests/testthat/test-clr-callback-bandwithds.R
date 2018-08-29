context("test-cyclical-learning-rate-callback-shifting-bandwidths")

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
    step_size = 13 * 6,
    base_lr = 0.001,
    max_lr = 0.006,
    factor = 0.8,
    patience = 3,
    cooldown = 5,
    decrease_base_lr = TRUE,
    verbose = 0
  )
  generate_model() %>% fit(
    train_data, train_targets,
    validation_data = list(test_data, test_targets),
    epochs = 50, verbose = 0,
    callbacks = list(callback_clr)
  )
  # plot_clr_history(callback_clr)
  expect_equal_to_reference(
    callback_clr$history,
    test_path("reference-objects/shifted-triangle-iteration")
  )
  expect_equal_to_reference(
    callback_clr$history_epoch,
    test_path("reference-objects/shifted-triangle-epoch")
  )
  saveRDS(callback_clr, test_path("reference-objects/callback-clr-for-plotting"))
})

test_that("triangle2", {
  callback_clr <- new_callback_cyclical_learning_rate(
    step_size = 13 * 6,
    base_lr = 0.001,
    max_lr = 0.006,
    factor = 0.4,
    patience = 5,
    decrease_base_lr = TRUE,
    verbose = 0, mode = "triangular2"
  )
  history <- generate_model() %>% fit(
    train_data, train_targets,
    validation_data = list(test_data, test_targets),
    epochs = 30, verbose = 0,
    callbacks = list(callback_clr)
  )
  plot(history)
  # plot_clr_history(callback_clr)
  expect_equal_to_reference(
    callback_clr$history,
    test_path("reference-objects/shifted-log-triangle2")
  )
})

