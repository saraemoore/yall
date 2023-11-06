#' Alternative Super Learner Prediction Function
#' 
#' This function is a refactored and stripped down version of
#' \code{\link[SuperLearner]{SampleSplitSuperLearner}}.
#' 
#' @param Y Numeric vector containing observed outcome values for the training
#' set.
#' @param X A \code{data.frame} containing predictor values for the training
#' set.
#' @param newX Optional \code{data.frame} containing predictor values for the
#' test set.
#' @param family A character string or function describing the error
#' distribution. Should be one of "gaussian" or "binomial". Defaults to
#' \code{\link{gaussian}}.
#' @param SL.library A character vector of prediction algorithm names or a list
#' containing these character vectors.
#' @param id Optional numeric vector containing cluster IDs.
#' @param verbose A boolean indicating whether diagnostic messages should be
#' printed. Defaults to \code{FALSE}.
#' @param saveFitLibrary Boolean indicating whether to retain the fit object for
#' each algorithm in the library in the returned result. Defaults to
#' \code{FALSE}.
#' @param obsWeights Optional numeric vector containing observation weights.
#' @return A list. See \code{\link[SuperLearner]{SampleSplitSuperLearner}}
#' for details.
#' @importFrom stats gaussian
fitPredictLibSL <- function(Y, X, newX = NULL, family = gaussian(), SL.library,
                            id = NULL, verbose = FALSE, saveFitLibrary = FALSE, obsWeights = NULL) {

    call <- match.call(expand.dots = TRUE)

    # put together the library
    # should this be in a new environment?
    .makeLibraryDF <- function(lib) {
        # pad each library algorithm
        library <- lapply(as.list(lib), function(x) {length(x) <- 2; return(x) })
        # combine into a matrix
        library <- do.call(rbind, as.list(library))
        colnames(library) <- c("predAlgorithm", "screenAlgorithm")
        # fill in screening algorithm as "All" if not specified
        library[is.na(library[,"screenAlgorithm"]), "screenAlgorithm"] <- "All"
        library <- as.data.frame(library, stringsAsFactors = FALSE)

        SuperLearner:::.check.SL.library(library = c(unique(library$predAlgorithm),
                                                     unique(library$screenAlgorithm)))
        return(library)
    }
    library <- .makeLibraryDF(SL.library)

    # should we be checking X and newX for data.frame?
    # data.frame not required, but most of the built-in wrappers assume a data.frame
    if(!inherits(X, 'data.frame')) message("X is not a data frame. Check the algorithms in SL.library ",
                                           "to make sure they are compatible with non data.frame inputs")
    varNames <- colnames(X)
    N <- dim(X)[1L]
    p <- dim(X)[2L]
    k <- nrow(library)
    kScreen <- length(unique(library$screenAlgorithm))
    libraryNames <- paste(library$predAlgorithm, library$screenAlgorithm, sep = "_")

    # put fitLibrary in its own environment to locate later
    fitLibEnv <- new.env()
    assign('fitLibrary', vector('list', length = k), envir = fitLibEnv)
    assign('libraryNames', libraryNames, envir = fitLibEnv)
    evalq(names(fitLibrary) <- libraryNames, envir = fitLibEnv)

    # errors* records if an algorithm stops either in the CV step and/or in full data
    errorsInCVLibrary <- rep(0, k)

    # if newX is missing, use X
    if(is.null(newX)) {
        newX <- X
    }
    # Are these checks still required?
    if(!identical(colnames(X), colnames(newX))) {
        stop("The variable names and order in newX must be identical to the variable names and order in X")
    }
    if (sum(is.na(X)) > 0 | sum(is.na(newX)) > 0 | sum(is.na(Y)) > 0) {
        stop("missing data is currently not supported. Check Y, X, and newX for missing values")
    }
    if (!is.numeric(Y)) {
        stop("the outcome Y must be a numeric vector")
    }

    # family can be either character or function, so these lines put everything together (code from glm())
    if(is.character(family))
        family <- get(family, mode="function", envir = parent.frame())
    if(is.function(family))
        family <- family()
    if (is.null(family$family)) {
        print(family)
        stop("'family' not recognized")
    }

    .checkLength <- function(x) {
        if(!identical(length(x[[1]]), length(x[[2]]))) {
            stop(paste(names(x)[1], "vector must have the same dimension as", names(x)[2]))
        }
    }
    # test id
    if(is.null(id)) {
        id <- seq(N)
    }
    .checkLength(list(id = id, Y = Y))
    # test observation weights
    if(is.null(obsWeights)) {
        obsWeights <- rep(1, N)
    }
    .checkLength(list(obsWeights = obsWeights, Y = Y))

    .screenFun <- function(fun, argList, isFullData = FALSE, verbose = FALSE) {
        testScreen <- try(do.call(fun, argList))
        out <- if(inherits(testScreen, "try-error")) {
            warning(paste("replacing failed screening algorithm,", fun, ", with All()",
                          ifelse(isFullData, " in full data", ""), "\n "))
            rep(TRUE, ncol(argList$X))
        } else {
            testScreen
        }
        if(verbose&!isFullData) {
            message(paste("Number of covariates in", fun, "is:", sum(out)))
        }
        return(out)
    }

    .predFun <- function(screenAlgorithm, predAlgorithm, Y, dataX, newX, whichScreen, family,
                         id, obsWeights, verbose, saveFitLibrary, isFullData = FALSE) {
        if(verbose) {
            libraryName <- paste(predAlgorithm, screenAlgorithm, sep = "_")
            message(paste(ifelse(isFullData, "full", "Fit"), libraryName))
        }
        testAlg <- try({
            select <- whichScreen[screenAlgorithm, ]
            if (all(select)) {
                tempX <- dataX
                tempnewX <- newX
            } else {
                tempX <- subset(dataX, select = which(select), drop=FALSE)
                tempnewX <- subset(newX, select = which(select), drop=FALSE)
            }
            do.call(predAlgorithm, list(
                Y = Y,
                X = tempX,
                newX = tempnewX,
                family = family, id = id, obsWeights = obsWeights))
        })

        out <- if(inherits(testAlg, "try-error")) {
            warning(paste("Error in algorithm", predAlgorithm, ifelse(isFullData, " on full data", ""),
                          "\n  The Algorithm will be removed from the Super Learner (i.e. given weight 0) \n" ))
            rep.int(NA, times = nrow(newX))
        } else {
            if(saveFitLibrary) {
                eval(bquote(fitLibrary[[.(1)]] <- .(testAlg$fit)), envir = fitLibEnv)
            }
            testAlg$pred
        }
        invisible(out)
    }

    whichScreen <- sapply(unique(library$screenAlgorithm), .screenFun,
                          argList = list(Y = Y, X = X, family = family, id = id, obsWeights = obsWeights),
                          verbose = verbose, simplify = FALSE)
    whichScreen <- do.call(rbind, whichScreen)
    colnames(whichScreen) <- colnames(X)

    predY <- mapply(.predFun, library$screenAlgorithm, library$predAlgorithm,
                    MoreArgs = list(Y = Y, dataX = X, newX = newX, whichScreen = whichScreen,
                                    family = family, id = id, obsWeights = obsWeights, verbose = verbose,
                                    saveFitLibrary = saveFitLibrary),
                    USE.NAMES = FALSE)
    # assume that a non-matrix response is a vector and should be a matrix with one row
    if(!is.matrix(predY)) {
        predY <- matrix(predY, ncol = nrow(library))
    }
    # add names of algorithms to the predictions
    colnames(predY) <- libraryNames

    errorsInCVLibrary <- apply(predY, 2, function(x) any(is.na(x)))
    if(sum(errorsInCVLibrary) > 0) {
        predY[, as.logical(errorsInCVLibrary)] <- 0
    }
    if(all(predY == 0)) {
        stop("All algorithms dropped from library")
    }

    # libRisk <- apply(predY, 2, function(yhat) mean(obsWeights * (yhat - Y)^2))
    # names(libRisk) <- libraryNames

    # put everything together in a list
    out <- list(libraryNames = libraryNames, SL.library = library, library.predict = predY,
                family = family, fitLibrary = get('fitLibrary', envir = fitLibEnv),
                varNames = varNames, whichScreen = whichScreen)
    return(out)
}

#' Wrapper function for \code{GridLearner}
#' 
#' @param X A \code{data.frame} of selected predictor values for only the
#' training set.
#' @param Y A numeric vector of outcome values for only the training set.
#' @param W A sparse vector of distances for only the current validation
#' observation with length equal to the number of rows in \code{X}.
#' @param newX A single-row \code{data.frame} of selected predictor values for
#' only the current validation observation.
#' @param newY A numeric outcome value for only the current validation
#' observation.
#' @param family A character string or function describing the error
#' distribution. Should be one of "gaussian" or "binomial".
#' @param SL.library A list containing character vectors where the first element
#' is a screening algorithm and the second is a prediction algorithm. See
#' \code{\link[SuperLearner]{SuperLearner}}'s \code{SL.library} argument for
#' more details.
#' @param control A list containing two named elements: "saveFitLibrary" (a
#' boolean indicating whether to retain the fit object for each algorithm in the
#' library in the returned result) and "trimLogit" (for the NNloglik combination
#' method, the level to which to truncate the logit transformation). See
#' \code{\link[SuperLearner]{SuperLearner.control}} for details.
#' @param verbose A boolean indicating whether diagnostic messages should be
#' printed. Defaults to \code{FALSE}.
#' @return A \code{list} of class \code{SuperLearner}.
SuperLearnerLOO <- function(X, Y, W, newX, newY, #newW,
                            family, SL.library, #method,
                            control, verbose = FALSE) {

    ssslres <- GridLearner(Y, X, newX = newX,
                           family = family, SL.library = SL.library,
                           verbose = verbose, control = control, obsWeights = W)

    ssslres$cvRisk <- (ssslres$library.predict[1,] - newY)^2
    colnames(ssslres$whichScreen) = ssslres$varNames
    return(ssslres)
}

#' For one validation set row and one set of parameters, estimate library predictions
#' 
#' @param X A \code{data.frame} of selected predictor values for the entire
#' tuning set.
#' @param W A sparse matrix of distances with number of rows equal to length of
#' \code{validRows} and the number of and number of columns equal to the number
#' of rows in \code{X} minus the length of \code{validRows} (number of training
#' set observations).
#' @param Y A numeric vector of outcome values for the entire tuning set.
#' @param validIdx Index within \code{validRows} representing the 'current'
#' validation observation.
#' @param validRows A numeric vector of row numbers within \code{X} (and indices
#' within \code{Y}) to use as a validation set within the current fold.
#' @param family A character string or function describing the error
#' distribution. Should be one of "gaussian" or "binomial".
#' @param SL.library A list containing character vectors where the first element
#' is a screening algorithm and the second is a prediction algorithm. See
#' \code{\link[SuperLearner]{SuperLearner}}'s \code{SL.library} argument for
#' more details.
#' @param control A list containing two named elements: "saveFitLibrary" (a
#' boolean indicating whether to retain the fit object for each algorithm in the
#' library in the returned result) and "trimLogit" (for the NNloglik combination
#' method, the level to which to truncate the logit transformation). See
#' \code{\link[SuperLearner]{SuperLearner.control}} for details.
#' @param verbose A boolean indicating whether diagnostic messages should be
#' printed. Defaults to \code{FALSE}.
#' @return A \code{data.frame}
#' @importFrom stats setNames
#' @importFrom broom tidy
#' @importFrom dplyr group_by do ungroup bind_cols full_join
#' @importFrom magrittr `%>%`
#' @importFrom tibble tibble
#' @import SLScreenExtra
SuperLearnerLOOValid <- function(X, W, Y, validIdx, validRows, family, SL.library, #method,
    control, verbose = FALSE) {
    # this index is in the original data.frame.
    validRow = validRows[validIdx]
    trainRows = seq_along(Y)[-validRows]

    # only want weights relevant for current validation set observation, not including that observation
    # newW = as.numeric(W[validIdx, validRow])
    # W = W[validIdx, -validRows]
    W = W[validIdx, ]

    newX = X[validRow, , drop = FALSE]
    X = X[-validRows, , drop = FALSE]

    newY = Y[validRows[validIdx]]
    Y = Y[-validRows]

    ssslres = SuperLearnerLOO(X = X, Y = Y, W = W, newX = newX, newY = newY, #newW = newW,
                              family = family, SL.library = SL.library, #method = method,
                              control = control, verbose = verbose)

    screenDF = tidy(ssslres, algorithm = "screening", includeAll = TRUE)
    screenDF = screenDF %>%
        group_by(screener) %>%
        do(keep2 = setNames(.$selected, .$term)) %>%
        ungroup() #%>%
        # mutate(keep2 = map(keep2, function(i) i[colnames(X)]))

    predDF = with(ssslres, bind_cols(tibble(library.predict = library.predict[1,],
                                            Z = as.vector(Z),
                                            # SL.predict = SL.predict[[1]],
                                            libraryNames = libraryNames,
                                            fitLibrary = unlist(fitLibrary, recursive = FALSE),
                                            obsNum = validRow),
                                     tidy(ssslres)))

    dplyr::full_join(predDF, screenDF, by = "screener")

    # SL.predict = setNames(SL.predict[[1]], as.character(validRow)),
    # whichTrainObs = trainRows[W > 0],
    # trainObsWeights = setNames(W, as.character(trainRows)),
    # validObsWeight = setNames(newW, as.character(validRow))))
}

#' For one validation set row and each set of parameters, estimate library predictions
#' 
#' @param validIdx Index within \code{validRows} representing the 'current'
#' validation observation.
#' @param validRows A numeric vector of row numbers within \code{X} (and indices
#' within \code{Y}) to use as a validation set within the current fold.
#' @param X A \code{data.frame} of predictor values for the entire tuning set.
#' @param Y A numeric vector of outcome values for the entire tuning set.
#' @param weightDF A \code{data.frame} containing at least columns "keep_names"
#' (where each value contains a character vector of retained features) and
#' "kernel_weights" (where each value contains a sparse matrix of distances with
#' number of rows equal to length of \code{validRows} and the number of and
#' number of columns equal to the number of rows in \code{X} minus the length of
#' \code{validRows}).
#' @param family A character string or function describing the error
#' distribution. Should be one of "gaussian" or "binomial".
#' @param SL.library A list containing character vectors where the first element
#' is a screening algorithm and the second is a prediction algorithm. See
#' \code{\link[SuperLearner]{SuperLearner}}'s \code{SL.library} argument for
#' more details.
#' @param control A list containing two named elements: "saveFitLibrary" (a
#' boolean indicating whether to retain the fit object for each algorithm in the
#' library in the returned result) and "trimLogit" (for the NNloglik combination
#' method, the level to which to truncate the logit transformation). See
#' \code{\link[SuperLearner]{SuperLearner.control}} for details.
#' @param verbose A boolean indicating whether diagnostic messages should be
#' printed. Defaults to \code{FALSE}.
#' @return A \code{data.frame}
#' @importFrom dplyr transmute do bind_rows slice n bind_cols
#' @importFrom magrittr `%>%`
#' @importFrom purrr map2
SuperLearnerLOOMultiValid <- function(validIdx, validRows, X, Y, weightDF, family, SL.library, #method,
                                      control, verbose = FALSE) {

    # kernel_weights and keep_names are columns of weightDF
    ssloores = weightDF %>%
        transmute(res = map2(lapply(keep_names, function(i) X[, i, drop = FALSE]),
                                    kernel_weights,
                                    SuperLearnerLOOValid,
                                    Y, validIdx, validRows, family, SL.library, #method,
                                    control, verbose = verbose)) %>%
        do(bind_rows(.$res))

    weightDF %>%
        slice(rep(1:n(), each = length(SL.library))) %>%
        bind_cols(ssloores)
}

#' For each validation set row and set of parameters, estimate library predictions
#' 
#' @param weightDF A \code{data.frame} containing at least columns "keep_names"
#' (where each value contains a character vector of retained features) and
#' "kernel_weights" (where each value contains a sparse matrix of distances with
#' number of rows equal to length of \code{validRows} and the number of and
#' number of columns equal to the number of rows in \code{X} minus the length of
#' \code{validRows}).
#' @param Y A numeric vector of outcome values for the entire tuning set.
#' @param X A \code{data.frame} of predictor values for the entire tuning set.
#' @param family A character string or function describing the error
#' distribution. Should be one of "gaussian" or "binomial".
#' @param SL.library A list containing character vectors where the first element
#' is a screening algorithm and the second is a prediction algorithm. See
#' \code{\link[SuperLearner]{SuperLearner}}'s \code{SL.library} argument for
#' more details.
#' @param control A list containing two named elements: "saveFitLibrary" (a
#' boolean indicating whether to retain the fit object for each algorithm in the
#' library in the returned result) and "trimLogit" (for the NNloglik combination
#' method, the level to which to truncate the logit transformation). See
#' \code{\link[SuperLearner]{SuperLearner.control}} for details.
#' @param validRows A numeric vector of row numbers within \code{X} (and
#' indices within \code{Y}) to use as a validation set within the current fold.
#' @param verbose A boolean indicating whether diagnostic messages should be
#' printed. Defaults to \code{FALSE}.
#' @return A \code{data.frame}
#' @importFrom tidyr unnest
#' @importFrom dplyr bind_rows
#' @importFrom magrittr `%>%`
SuperLearnerLOOMultiAllValid <- function(weightDF, Y, X, family, SL.library, #method,
                                         control, validRows, verbose = FALSE) {

    ssloores = lapply(seq_along(validRows), SuperLearnerLOOMultiValid, validRows, X, Y, weightDF,
                      family, SL.library, control, verbose = verbose) %>%
                bind_rows()

    # expand combo_method, selector, k
    ssloores %>%
        unnest(combo_method, selector, k, .drop = FALSE)
}