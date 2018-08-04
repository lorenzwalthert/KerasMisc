#' Initiate a new cyclical learning rate scheduler
#'
#' This callback implements a cyclical learning rate policy (CLR).
#' The method cycles the learning rate between two boundaries with
#' some constant frequency, as detailed in
#' [this paper](https://arxiv.org/abs/1506.01186).
#'
#' @details
#' The amplitude of the cycle can be scaled on a per-iteration or per-cycle
#' basis. This class has three built-in policies, as put forth in the paper.
#'
#' - "triangular": A basic triangular cycle w/ no amplitude scaling.
#' - "triangular2": A basic triangular cycle that scales initial amplitude by
#'   half each cycle.
#' - "exp_range": A cycle that scales initial amplitude by gamma**(cycle
#'   iterations) at each cycle iteration.
#'
#' For more details, please see paper.
#' @param base_lr Initial learning rate which is the lower boundary in the
#'   lower boundary in the cycle.
#' @param max_lr Upper boundary in the cycle. Functionally, it defines the
#'    cycle amplitude (`max_lr - base_lr`). The learning rate at any cycle is
#'    the sum of `base_lr` and some scaling of the amplitude; therefore `max_lr`
#'    may not actually be reached depending on scaling function.
#' @param step_size Number of training iterations per half cycle. Authors
#'   suggest setting step_size `2-8` x training iterations in epoch.
#' @param mode One of "triangular", "triangular2" or "exp_range". Default
#'   "triangular". Values correspond to policies detailed above. If `scale_fn`
#'   is not `NULL`, this argument is ignored.
#' @param gamma Constant in `exp_range` scaling function: `gamma^(cycle iterations)`
#' @param scale_fn Custom scaling policy defined by a single argument anonymous
#'   function, where `0 <= scale_fn(x) <= 1` for all `x >= 0`. Mode paramater is
#'   ignored.
#' @param scale_mode Either "cycle" or "iterations". Defines whether `scale_fn`
#'   is evaluated on cycle number or cycle iterations (training iterations since
#'   start of cycle). Default is "cycle".
#' @family callbacks
#' @export
#' @example
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
new_callback_cyclical_learning_rate <- function(base_lr = 0.001,                                                 max_lr = 0.006,
                                                step_size = 2000,
                                                mode = "triangular",
                                                gamma = 1,
                                                scale_fn = NULL,
                                                scale_mode = "cycle") {
  CyclicLR$new(
    base_lr = base_lr,
    max_lr = max_lr,
    step_size = step_size,
    mode = mode,
    gamma = gamma,
    scale_fn = scale_fn,
    scale_mode = scale_mode
  )
}

#' @importFrom utils write.table
CyclicLR <- R6::R6Class("CyclicLR",
  inherit = keras::KerasCallback,
  public = list(
    base_lr = NULL,
    max_lr = NULL,
    step_size = NULL,
    mode = NULL,
    gamma = NULL,
    scale_fn = NULL,
    scale_mode = NULL,
    clr_iterations = NULL,
    trn_iterations = NULL,
    history = NULL,
    trn_epochs = NULL,

    initialize = function(base_lr = 0.001,
                          max_lr = 0.006,
                          step_size = 2000,
                          mode = "triangular",
                          gamma = 1,
                          scale_fn = NULL,
                          scale_mode = "cycle") {

      assert_CyclicLR_init(
        base_lr,
        max_lr,
        step_size,
        mode,
        gamma,
        scale_fn,
        scale_mode
      )

      self$base_lr <- base_lr
      self$max_lr <- max_lr
      self$step_size <- step_size
      self$mode <- mode
      self$gamma <- gamma

      if (is.null(scale_fn)) {
        if (self$mode == "triangular") {
          self$scale_fn <- function(x) 1
          self$scale_mode <- "cycle"
        } else if (self$mode == "triangular2") {
          self$scale_fn <- function(x) 1 / (2^(x - 1))
          self$scale_mode <- "cycle"
        } else if (self$mode == "exp_range") {
          self$scale_fn <- function(x) gamma^(x)
          self$scale_mode <- "iterations"
        }
      } else {
        self$scale_fn <- scale_fn
        self$scale_mode <- scale_mode
      }
      self$clr_iterations <- 0
      self$trn_iterations <- 0
      self$trn_epochs <- 1
      self$history <- data.frame()
      self$.reset()
    },

    .reset = function(new_base_lr = NULL,
                      new_max_lr = NULL,
                      new_step_size = NULL) {

      if (!is.null(new_base_lr)) {
        self$base_lr <- new_base_lr
      }
      if (!is.null(new_max_lr)) {
        self$max_lr <- new_max_lr
      }
      if (!is.null(new_step_size)) {
        self$step_size <- new_step_size
      }
      self$clr_iterations <- 0
    },

    clr = function() {
      cycle <- floor(1 + self$clr_iterations / (2 * self$step_size))
      x <- abs(self$clr_iterations / self$step_size - 2 * cycle + 1)
      if (self$scale_mode == "cycle") {
        self$base_lr +
          (self$max_lr - self$base_lr) *
          max(0, (1 - x)) *
          self$scale_fn(cycle)

      } else {
        self$base_lr +
          (self$max_lr - self$base_lr) * max(0, (1 - x)) * self$scale_fn(self$clr_iterations)
      }
    },

    on_train_begin = function(logs) {
      if (self$clr_iterations == 0) {
        k_set_value(self$model$optimizer$lr, self$base_lr)
      } else {
        k_set_value(self$model$optimizer$lr, self$clr())
      }
    },
    on_batch_end = function(batch, logs = list()) {
      new_history <- data.frame(
        lr = k_get_value(self$model$optimizer$lr),
        iterations = self$trn_iterations,
        epochs = self$trn_epochs
      )
      self$history <- rbind(
        self$history, new_history
      )
      self$trn_iterations <- self$trn_iterations + 1
      self$clr_iterations <- self$clr_iterations + 1
      k_set_value(self$model$optimizer$lr, self$clr())
    },
    on_epoch_end = function(epochs, logs) {
      self$trn_epochs = self$trn_epochs + 1
    }
  )
)


assert_CyclicLR_init <- function(
  base_lr,
  max_lr,
  step_size,
  mode,
  gamma,
  scale_fn,
  scale_mode
) {

  checkmate::assert_numeric(max_lr - base_lr, lower = 0)
  checkmate::assert_integerish(step_size, lower = 1)
  checkmate::assert_numeric(gamma)

  if (is.null(scale_fn)) {
    checkmate::assert_choice(mode,
      choices = c("triangular", "triangular2", "exp_range")
    )
  } else {
    checkmate::assert_function(scale_fn)
  }
  checkmate::assert_choice(scale_mode, choices = c("cycle", "iterations"))
}
