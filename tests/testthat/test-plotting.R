context("test-plotting")
library(vdiffr)
test_that("plotting works", {
  callback_clr <- readRDS(test_path("reference-objects/callback-clr-for-plotting"))
  expect_doppelganger("epoch-ggplot2", plot_clr_history(callback_clr))
  expect_doppelganger(
    "iterations-ggplot2",
    plot_clr_history(callback_clr, granularity = "iteration")
  )
  expect_doppelganger(
    "epoch-base",
    plot_clr_history(callback_clr, backend = "base")
  )
  expect_doppelganger("iterations-base", plot_clr_history(callback_clr,
    granularity = "iteration", backend = "base"
  ))
})
