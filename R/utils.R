log_to_csv <- function(self, initialize = FALSE, sep = ",") {
  browser()
  write.table(data.frame(
    lr = self$lr,
    current_batch_count = self$current_batch_count,
    current_epoch_count = self$current_epoch_count
  ),
  "lr.csv",
  append = !initialize, sep = sep,
  col.names = initialize, row.names = FALSE, quote = FALSE
  )
}
