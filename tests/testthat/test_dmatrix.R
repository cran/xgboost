require(xgboost)
require(Matrix)

context("testing xgb.DMatrix functionality")

data(agaricus.test, package = 'xgboost')
test_data <- agaricus.test$data[1:100, ]
test_label <- agaricus.test$label[1:100]

n_threads <- 2

test_that("xgb.DMatrix: basic construction", {
  # from sparse matrix
  dtest1 <- xgb.DMatrix(test_data, label = test_label, nthread = n_threads)

  # from dense matrix
  dtest2 <- xgb.DMatrix(as.matrix(test_data), label = test_label, nthread = n_threads)
  expect_equal(getinfo(dtest1, "label"), getinfo(dtest2, "label"))
  expect_equal(dim(dtest1), dim(dtest2))

  #from dense integer matrix
  int_data <- as.matrix(test_data)
  storage.mode(int_data) <- "integer"
  dtest3 <- xgb.DMatrix(int_data, label = test_label, nthread = n_threads)
  expect_equal(dim(dtest1), dim(dtest3))
})

test_that("xgb.DMatrix: NA", {
  n_samples <- 3
  x <- cbind(
    x1 = sample(x = 4, size = n_samples, replace = TRUE),
    x2 = sample(x = 4, size = n_samples, replace = TRUE)
  )
  x[1, "x1"] <- NA

  m <- xgb.DMatrix(x, nthread = n_threads)
  xgb.DMatrix.save(m, "int.dmatrix")

  x <- matrix(as.numeric(x), nrow = n_samples, ncol = 2)
  colnames(x) <- c("x1", "x2")
  m <- xgb.DMatrix(x, nthread = n_threads)

  xgb.DMatrix.save(m, "float.dmatrix")

  iconn <- file("int.dmatrix", "rb")
  fconn <- file("float.dmatrix", "rb")

  expect_equal(file.size("int.dmatrix"), file.size("float.dmatrix"))

  bytes <- file.size("int.dmatrix")
  idmatrix <- readBin(iconn, "raw", n = bytes)
  fdmatrix <- readBin(fconn, "raw", n = bytes)

  expect_equal(length(idmatrix), length(fdmatrix))
  expect_equal(idmatrix, fdmatrix)

  close(iconn)
  close(fconn)

  file.remove("int.dmatrix")
  file.remove("float.dmatrix")
})

test_that("xgb.DMatrix: saving, loading", {
  # save to a local file
  dtest1 <- xgb.DMatrix(test_data, label = test_label, nthread = n_threads)
  tmp_file <- tempfile('xgb.DMatrix_')
  on.exit(unlink(tmp_file))
  expect_true(xgb.DMatrix.save(dtest1, tmp_file))
  # read from a local file
  expect_output(dtest3 <- xgb.DMatrix(tmp_file), "entries loaded from")
  expect_output(dtest3 <- xgb.DMatrix(tmp_file, silent = TRUE), NA)
  unlink(tmp_file)
  expect_equal(getinfo(dtest1, 'label'), getinfo(dtest3, 'label'))

  # from a libsvm text file
  tmp <- c("0 1:1 2:1", "1 3:1", "0 1:1")
  tmp_file <- 'tmp.libsvm'
  writeLines(tmp, tmp_file)
  expect_true(file.exists(tmp_file))
  dtest4 <- xgb.DMatrix(
    paste(tmp_file, "?format=libsvm", sep = ""), silent = TRUE, nthread = n_threads
  )
  expect_equal(dim(dtest4), c(3, 4))
  expect_equal(getinfo(dtest4, 'label'), c(0, 1, 0))

  # check that feature info is saved
  data(agaricus.train, package = 'xgboost')
  dtrain <- xgb.DMatrix(
    data = agaricus.train$data, label = agaricus.train$label, nthread = n_threads
  )
  cnames <- colnames(dtrain)
  expect_equal(length(cnames), 126)
  tmp_file <- tempfile('xgb.DMatrix_')
  xgb.DMatrix.save(dtrain, tmp_file)
  dtrain <- xgb.DMatrix(tmp_file)
  expect_equal(colnames(dtrain), cnames)

  ft <- rep(c("c", "q"), each=length(cnames)/2)
  setinfo(dtrain, "feature_type", ft)
  expect_equal(ft, getinfo(dtrain, "feature_type"))
})

test_that("xgb.DMatrix: getinfo & setinfo", {
  dtest <- xgb.DMatrix(test_data, nthread = n_threads)
  expect_true(setinfo(dtest, 'label', test_label))
  labels <- getinfo(dtest, 'label')
  expect_equal(test_label, getinfo(dtest, 'label'))

  expect_true(setinfo(dtest, 'label_lower_bound', test_label))
  expect_equal(test_label, getinfo(dtest, 'label_lower_bound'))

  expect_true(setinfo(dtest, 'label_upper_bound', test_label))
  expect_equal(test_label, getinfo(dtest, 'label_upper_bound'))

  expect_true(length(getinfo(dtest, 'weight')) == 0)
  expect_true(length(getinfo(dtest, 'base_margin')) == 0)

  expect_true(setinfo(dtest, 'weight', test_label))
  expect_true(setinfo(dtest, 'base_margin', test_label))
  expect_true(setinfo(dtest, 'group', c(50, 50)))
  expect_error(setinfo(dtest, 'group', test_label))

  # providing character values will give an error
  expect_error(setinfo(dtest, 'weight', rep('a', nrow(test_data))))

  # any other label should error
  expect_error(setinfo(dtest, 'asdf', test_label))
})

test_that("xgb.DMatrix: slice, dim", {
  dtest <- xgb.DMatrix(test_data, label = test_label, nthread = n_threads)
  expect_equal(dim(dtest), dim(test_data))
  dsub1 <- slice(dtest, 1:42)
  expect_equal(nrow(dsub1), 42)
  expect_equal(ncol(dsub1), ncol(test_data))

  dsub2 <- dtest[1:42, ]
  expect_equal(dim(dtest), dim(test_data))
  expect_equal(getinfo(dsub1, 'label'), getinfo(dsub2, 'label'))
})

test_that("xgb.DMatrix: slice, trailing empty rows", {
  data(agaricus.train, package = 'xgboost')
  train_data <- agaricus.train$data
  train_label <- agaricus.train$label
  dtrain <- xgb.DMatrix(
    data = train_data, label = train_label, nthread = n_threads
  )
  slice(dtrain, 6513L)
  train_data[6513, ] <- 0
  dtrain <- xgb.DMatrix(
    data = train_data, label = train_label, nthread = n_threads
  )
  slice(dtrain, 6513L)
  expect_equal(nrow(dtrain), 6513)
})

test_that("xgb.DMatrix: colnames", {
  dtest <- xgb.DMatrix(test_data, label = test_label, nthread = n_threads)
  expect_equal(colnames(dtest), colnames(test_data))
  expect_error(colnames(dtest) <- 'asdf')
  new_names <- make.names(seq_len(ncol(test_data)))
  expect_silent(colnames(dtest) <- new_names)
  expect_equal(colnames(dtest), new_names)
  expect_silent(colnames(dtest) <- NULL)
  expect_null(colnames(dtest))
})

test_that("xgb.DMatrix: nrow is correct for a very sparse matrix", {
  set.seed(123)
  nr <- 1000
  x <- rsparsematrix(nr, 100, density = 0.0005)
  # we want it very sparse, so that last rows are empty
  expect_lt(max(x@i), nr)
  dtest <- xgb.DMatrix(x, nthread = n_threads)
  expect_equal(dim(dtest), dim(x))
})
