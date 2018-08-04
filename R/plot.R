#' Simple plotting utility
#'
#' @param callback_clr An object of class `CyclicLR`.
#' @importFrom graphics plot
#' @examples
#' library(keras)
#' dataset <- dataset_boston_housing()
#' c(c(train_data, train_targets), c(test_data, test_targets)) %<-% dataset
#'
#' mean <- apply(train_data, 2, mean)
#' std <- apply(train_data, 2, sd)
#' train_data <- scale(train_data, center = mean, scale = std)
#' test_data <- scale(test_data, center = mean, scale = std)
#'
#'
#' model <- keras_model_sequential() %>%
#'   layer_dense(
#'     units = 64, activation = "relu",
#'     input_shape = dim(train_data)[[2]]
#'   ) %>%
#'   layer_dense(units = 64, activation = "relu") %>%
#'   layer_dense(units = 1)
#' model %>% compile(
#'   optimizer = optimizer_rmsprop(lr = 0.001),
#'   loss = "mse",
#'   metrics = c("mae")
#' )
#'
#' callback_clr <- new_callback_cyclical_learning_rate(
#'   step_size = 32,
#'   base_lr = 0.001,
#'   max_lr = 0.006,
#'   gamma = 0.99,
#'   mode = "exp_range"
#' )
#' model %>% fit(
#'   train_data, train_targets,
#'   validation_data = list(test_data, test_targets),
#'   epochs = 10, verbose = 1,
#'   callbacks = list(callback_clr)
#' )
#' callback_clr$history
#' plot_clr_history(callback_clr)
#' @export
plot_clr_history <- function(callback_clr) {
  checkmate::assert_class(callback_clr, "CyclicLR")
  history <- callback_clr$history
  plot(history$iterations, history$lr, col = history$epochs)
}
