#' Initiate a new cyclical learning rate scheduler
#'
#' This callback implements a cyclical learning rate policy (CLR).
#' The method cycles the learning rate between two boundaries with
#' some constant frequency, as detailed in
#' [this paper](https://arxiv.org/abs/1506.01186). In addition, the
#' call-back supports scaled learning-rate bandwidths (see section
#' 'Differences to the Python implementation'). Note that this callback is
#' very general as it can be used to specify:
#'
#' * constant learning rates.
#' * cyclical learning rates.
#' * decayling learning rates depending on validation loss such as
#'   [keras::reduce_lr_on_plateau()]
#' * learning rates with scaled bandwidths.
#' Apart from this, the
#' implementation follows the
#' [Python implementation](https://github.com/bckenstler/CLR) quite closey.
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
#' @section Differences to Python implementation:
#' This implementation differs from the
#' [Python implementation](https://github.com/bckenstler/CLR) in the following
#' aspects:
#'
#' - *scaled learning-rate bandwidth on plateau* is supported. Via the
#'   arguments `patience`, `factor` and `decrease_base_lr`, the user has
#'   control over if and when the boundaries of the learning rate are adjusted.
#'   This feature allows to combine decaying learning rates with cyclical
#'   learning rates. Typically, one wants to reduce the learning rate bandwith
#'   after validation loss has stopped improving for some time.
#'   Note that both `factor < 1` and `patience < Inf` must hold
#'   in order for this feature to take effect.
#' - The `history` dataframe in the return value of this callback has a column
#'   `epochs` in addition to `itterations` and `lr`.
#' - The callback returns a `history_epoch` dataframe that just contains the
#'   epochs and the learning rates at the end of the epoch. This is less
#'   granular than the `history` element.
#' - All column names in `history` and `history_epoch` are - opposed to the
#'   Python implementation - in singular.
#' @param base_lr Initial learning rate which is the lower boundary in the
#'   cycle.
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
#' @param patience The number of epochs of training without validation loss
#'   improvement that the callback will wait before it adjusts `base_lr` and
#'   `max_lr`.
#' @param factor An numeric vector of lenght one which will scale `max_lr` and
#'   (if applicable according to `decrease_base_lr`) `base_lr`
#'   after `patience` epochs without improvement in the validation loss.
#' @param decrease_base_lr Boolean indicating whether `base_lr` should also be
#'   scaled with `factor` or not.
#' @family callbacks
#' @export
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
new_callback_cyclical_learning_rate <- function(base_lr = 0.001,                                                 max_lr = 0.006,
                                                step_size = 2000,
                                                mode = "triangular",
                                                gamma = 1,
                                                scale_fn = NULL,
                                                scale_mode = "cycle",
                                                patience = Inf,
                                                factor = 0.9,
                                                decrease_base_lr = TRUE,
                                                cooldown = 2,
                                                verbose = 1
                                                ) {
  CyclicLR$new(
    base_lr = base_lr,
    max_lr = max_lr,
    step_size = step_size,
    mode = mode,
    gamma = gamma,
    scale_fn = scale_fn,
    scale_mode = scale_mode,
    patience = patience,
    factor = factor,
    decrease_base_lr = decrease_base_lr,
    verbose = verbose,
    cooldown = cooldown
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
    patience = NULL,
    factor = NULL,
    clr_iteration = NULL,
    trn_iteration = NULL,
    history = NULL,
    history_epoch = NULL,
    trn_epochs = NULL,
    not_improved_for_n_times = NULL,
    decrease_base_lr = NULL,
    verbose = NULL,
    cooldown = NULL,
    cooldown_counter = NULL,
    initialize = function(base_lr = 0.001,
                          max_lr = 0.006,
                          step_size = 2000,
                          mode = "triangular",
                          gamma = 1,
                          scale_fn = NULL,
                          scale_mode = "cycle",
                          patience = 3,
                          factor = 0.9,
                          decrease_base_lr = TRUE,
                          cooldown = 0,
                          verbose = 1) {

      assert_CyclicLR_init(
        base_lr,
        max_lr,
        step_size,
        mode,
        gamma,
        scale_fn,
        scale_mode,
        patience,
        factor,
        decrease_base_lr,
        cooldown,
        verbose
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
          self$scale_mode <- "iteration"
        }
      } else {
        self$scale_fn <- scale_fn
        self$scale_mode <- scale_mode
      }
      self$clr_iteration <- 0
      self$trn_iteration <- 0
      self$trn_epochs <- 1
      self$history <- data.frame()
      self$history_epoch <- data.frame()
      self$patience <- patience
      self$factor <- factor
      self$decrease_base_lr <- decrease_base_lr
      self$cooldown <- cooldown
      self$cooldown_counter <- 0
      self$verbose <- verbose
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
      self$clr_iteration <- 0
      self$not_improved_for_n_times <- 0
    },
    clr = function() {
      cycle <- floor(1 + self$clr_iteration / (2 * self$step_size))
      x <- abs(self$clr_iteration / self$step_size - 2 * cycle + 1)
      if (self$scale_mode == "cycle") {
        self$base_lr +
          (self$max_lr - self$base_lr) *
          max(0, (1 - x)) *
          self$scale_fn(cycle)

      } else {
        self$base_lr +
          (self$max_lr - self$base_lr) * max(0, (1 - x)) * self$scale_fn(self$clr_iteration)
      }
    },
    in_cooldown = function() {
      self$cooldown_counter > 0
    },

    on_train_begin = function(logs) {
      if (self$clr_iteration == 0) {
        k_set_value(self$model$optimizer$lr, self$base_lr)
      } else {
        k_set_value(self$model$optimizer$lr, self$clr())
      }
    },
    on_batch_end = function(batch, logs = list()) {
      new_history <- data.frame(
        lr = k_get_value(self$model$optimizer$lr),
        base_lr = self$base_lr,
        max_lr = self$max_lr,
        iteration = self$trn_iteration,
        epochs = self$trn_epochs
      )
      self$history <- rbind(
        self$history, new_history
      )
      self$trn_iteration <- self$trn_iteration + 1
      self$clr_iteration <- self$clr_iteration + 1
      k_set_value(self$model$optimizer$lr, self$clr())
    },
    on_epoch_end = function(epochs, logs = list()) {
      self$trn_epochs = self$trn_epochs + 1
      best <- ifelse(nrow(self$history_epoch) > 0,
                     min(self$history_epoch$val_loss),
                     logs$val_loss
      )
      if(logs$val_loss > best) {
        self$not_improved_for_n_times <- self$not_improved_for_n_times + 1
      } else {
        self$not_improved_for_n_times <- 0
      }
      self$history_epoch <- rbind(
        self$history_epoch,
        as.data.frame(do.call(cbind, c(
          logs, epoch = epochs,
          not_improved_for_n_times = self$not_improved_for_n_times,
          base_lr = self$base_lr, max_lr = self$max_lr,
          lr = k_get_value(self$model$optimizer$lr),
          in_cooldown = self$in_cooldown()
        )))
      )
      if (self$in_cooldown()) {
        self$cooldown_counter <- self$cooldown_counter - 1L
        if (self$verbose > 0) {
          cat("cooling down.\n")
        }
      } else if (self$not_improved_for_n_times > self$patience) {
        if (self$decrease_base_lr) {
          if (self$verbose > 0) {
            cat("Adjusting base_lr to ", self$base_lr, ".\n", sep = "")
          }
          self$base_lr <-  self$factor * self$base_lr
          self$cooldown_counter <- self$cooldown
        }
        candidate_max_lr <- self$factor * self$max_lr
        if (self$base_lr <= candidate_max_lr) {
          self$max_lr <- candidate_max_lr
          if (self$verbose > 0) {
            cat("Adjusting max_lr to ", self$max_lr, ".\n", sep = "")
          }
        } else if (self$verbose > 0) {
          cat(
            "Can't adjust max_lr below base_lr. `Reduce` base_lr or ",
            "set `decrease_base_lr` when initializing the callback.\n",
            sep = ""
          )
        }
      }
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
  scale_mode,
  patience,
  factor,
  decrease_base_lr,
  cooldown,
  verbose
) {

  checkmate::assert_numeric(max_lr - base_lr, lower = 0)
  checkmate::assert_integerish(step_size, lower = 1)
  checkmate::assert_numeric(gamma)
  checkmate::assert_integerish(patience)
  checkmate::assert_numeric(factor)
  checkmate::assert_logical(decrease_base_lr)
  checkmate::assert_integerish(cooldown, lower = 0)
  checkmate::assert_integerish(verbose, lower = 0)
  if (is.null(scale_fn)) {
    checkmate::assert_choice(mode,
      choices = c("triangular", "triangular2", "exp_range")
    )
  } else {
    checkmate::assert_function(scale_fn)
  }
}
