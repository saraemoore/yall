#' Train the local learning algorithm on a single CV fold
#' 
#' \code{trainYALLfold} is the main interface to training the local learning
#' algorithm within a single cross-validation fold.
#' 
#' @param y_all A numeric vector of outcome values for the entire
#' tuning set.
#' @param x_all A \code{data.frame} of predictor values for the entire
#' tuning set.
#' @param fold_num A numeric value specifying the CV fold ID.
#' @param trainRows A numeric vector of row numbers within \code{x_all} (and
#' indices within \code{y_all}) to use for training for \code{fold_num}.
#' @param validRows A numeric vector of row numbers within \code{x_all} (and
#' indices within \code{y_all}) to use for validation for \code{fold_num}.
#' @param family A character string or function describing the error
#' distribution. Should be one of "gaussian" or "binomial".
#' @param cvSLfeatsel_control A named list containing tuning options for
#' \code{\link[SuperSelector]{cvSLFeatureSelector}}: "method", "SL.library",
#' "selector.library", "nFolds".
#' @param neighbor_control A named list containing a single tuning option
#' element for \code{link{distKernelWeights}}: "library". See the
#' \code{neighbor_library} argument of \code{link{distKernelWeights}} for
#' details.
#' @param predict_control A named list containing tuning options passed to
#' \code{\link{GridLearner}}: "SL.library" and "trimLogit".
#' @param save_dist A boolean indicating whether to retain the distance values
#' in the result. Passed to \code{\link{distKernelWeights}}. Defaults to
#' \code{FALSE}.
#' @param save_fit_library A boolean passed to \code{\link{GridLearner}} to
#' indicate whether to retain the fit object for each algorithm in the library
#' in the returned result. Defaults to \code{TRUE}. Note that a \code{TRUE}
#' setting can result in a very large returned result.
#' @param model_keep If \code{save_fit_library} is true, a character vector
#' indicating which, if any, "disposable" elements of the model fit objects
#' to retain. Defaults to \code{"everything"}. See \code{\link[strip]{strip}}
#' for details.
#' @param verbose A boolean indicating whether diagnostic messages should be
#' printed. Defaults to \code{FALSE}.
#' @return A named list with elements "predict", "screen", "screen_full", and
#' "screen_which".
#' @importFrom magrittr `%>%`
#' @importFrom dplyr transmute do bind_rows slice n bind_cols mutate
#' @importFrom purrr map
#' @importFrom tibble as_tibble
#' @importFrom strip strip
#' @importFrom SuperSelector cvSLFeatureSelector groupBySelectionSet
#' @export
trainYALLfold <- function(y_all, x_all, fold_num, trainRows, validRows, family,
                          cvSLfeatsel_control, neighbor_control, predict_control,
                          save_dist = FALSE, save_fit_library = TRUE,
                          model_keep = c("everything"), verbose = FALSE) {
    if(verbose) {
        cat("Global feature selection\n")
    }
    cvSLres <- cvSLFeatureSelector(y_all[trainRows], x_all[trainRows,],
                                   family = family,
                                   method = cvSLfeatsel_control$method,
                                   SL.library = cvSLfeatsel_control$SL.library,
                                   selector.library = cvSLfeatsel_control$selector.library,
                                   nFolds = cvSLfeatsel_control$nFolds,
                                   weighted = FALSE,
                                   verbose = verbose,
                                   label = c(metafold = fold_num))

    cvSLkeepUnique <- groupBySelectionSet(cvSLres)

    # run distKernelWeights() on each row of cvSLkeepUnique
    # this returns weights only for the training set, but SuperLearnerLOOMultiAllValid()
    #   expects weights for all observations (training and validation)
    obsWeightsDF = cvSLkeepUnique %>%
        transmute(res = map(keep_names, distKernelWeights, x_all[trainRows,], x_all[validRows,],
                            neighbor_control$library, save_dist = save_dist, verbose = verbose)) %>%
        do(bind_rows(.$res))

    obsWeightsDF = cvSLkeepUnique %>%
        slice(rep(1:n(), each = nrow(neighbor_control$library))) %>%
        bind_cols(obsWeightsDF)

    if(verbose) {
        cat("Prediction\n")
    }
    predres <- SuperLearnerLOOMultiAllValid(obsWeightsDF, y_all, x_all, family, predict_control$SL.library,
                                            control = list(saveFitLibrary = save_fit_library,
                                                           trimLogit = predict_control$trimLogit),
                                            validRows, verbose = verbose)

    if(!("error"%in%names(predres))) {
        if(save_fit_library) {
            predres <- predres %>% mutate(fitLibrary = map(fitLibrary, strip::strip, keep = model_keep))
        } else {
            predres$fitLibrary <- NULL
        }

        # label with fold number
        predres$fold = fold_num
    }

    return(list(predict = predres,
                screen = as_tibble(cvSLres$summary),
                screen_full = cvSLres$cvslFull,
                screen_which = cvSLres$whichVariable))
                # control = list(screen = cvSLfeatsel_control,
                               # neighbor = neighbor_control,
                               # predict = predict_control)))
}

#' Wrapper function for \code{trainYALLfold}
#' 
#' \code{trainYALL} is the main interface to training the local learning
#' algorithm over cross-validation folds.
#' 
#' @param fold One element of a list of folds created via
#' \code{\link[origami]{make_folds}}.
#' @param y_all A numeric vector of outcome values for the entire
#' tuning/training set.
#' @param x_all A \code{data.frame} of predictor values for the entire
#' tuning/training set.
#' @param family A character string or function describing the error
#' distribution. Should be one of "gaussian" or "binomial".
#' @param cvSLfeatsel_control A named list containing tuning options for
#' \code{\link[SuperSelector]{cvSLFeatureSelector}}: "method", "SL.library",
#' "selector.library", "nFolds".
#' @param neighbor_control A named list containing a single tuning option
#' element for \code{link{distKernelWeights}}: "library". See the
#' \code{neighbor_library} argument of \code{link{distKernelWeights}} for
#' details.
#' @param predict_control A named list containing tuning options passed to
#' \code{\link{GridLearner}}: "SL.library" and "trimLogit".
#' @param save_dist A boolean indicating whether to retain the distance values
#' in the result. Passed to \code{\link{distKernelWeights}}. Defaults to
#' \code{FALSE}.
#' @param save_fit_library A boolean passed to \code{\link{GridLearner}} to
#' indicate whether to retain the fit object for each algorithm in the library
#' in the returned result. Defaults to \code{TRUE}.
#' @param model_keep If \code{save_fit_library} is true, a character vector
#' indicating which, if any, "disposable" elements of the model fit objects
#' to retain. Defaults to \code{"everything"}. See \code{\link[strip]{strip}}
#' for details.
#' @param verbose A boolean indicating whether diagnostic messages should be
#' printed. Defaults to \code{FALSE}.
#' @return A list with length equal to the number of folds, with each list
#' element containing a named list with elements "predict", "screen",
#' "screen_full", and "screen_which".
#' @importFrom origami training validation
#' @export
trainYALL <- function(fold, y_all, x_all, family, cvSLfeatsel_control, neighbor_control, predict_control,
                      save_dist = FALSE, save_fit_library = TRUE, model_keep = c("everything"), verbose = FALSE) {

    trainRows <- origami::training()
    validRows <- origami::validation()

    ############################################################################

    trainYALLfold(y_all, x_all, fold$v, trainRows, validRows, family,
                  cvSLfeatsel_control, neighbor_control, predict_control,
                  save_dist = save_dist, save_fit_library = save_fit_library,
                  model_keep = model_keep, verbose = verbose)
}

#' Use a single set of tuning parameters to predict outcomes for a single record in the holdout set
#' 
#' @param newX A single-row \code{data.frame} of predictor values for one record
#' in the holdout set.
#' @param y_all A numeric vector of outcome values for the entire tuning set.
#' @param x_all A \code{data.frame} of predictor values for the entire tuning
#' set.
#' @param family A character string or function describing the error
#' distribution. Should be one of "gaussian" or "binomial".
#' @param neighbor_control A named list with one element "library" containing a
#' single-row \code{data.frame} with columns "norm", "kernel", and "window"
#' specifying tuning parameters to be used when calling
#' \code{\link{distKernelWeights}}.
#' @param predict_control A named list with one element "SL.library" containing
#' a single-element list, which in turn contains a character vector listing the
#' prediction and screening functions to be used when calling
#' \code{fitPredictLibSL}.
#' @param keepX A character vector of features in \code{x_all} and \code{newX}
#' to retain.
#' @param screenRow The "whichVariable" element of the result from calling
#' \code{\link[SuperSelector]{cvSLFeatureSelector}}.
#' @param verbose A boolean indicating whether diagnostic messages should be
#' printed. Defaults to \code{FALSE}.
#' @return A \code{data.frame}
#' @importFrom stats setNames
#' @importFrom tibble as_tibble tibble
#' @importFrom dplyr bind_cols group_by ungroup full_join do
#' @import SLScreenExtra
predictOneYALL <- function(newX, y_all, x_all, family, neighbor_control,
                           predict_control, keepX, screenRow, verbose = FALSE) {

    if(nrow(newX) > 1) {
        stop("Only one newX")
    }
    if(nrow(neighbor_control$library) > 1) {
        stop("Only one weight/kernel")
    }

    obsWeightsDF = distKernelWeights(keepX, x_all, newX, neighbor_control$library, verbose = verbose)
    obsWeightsDF = bind_cols(screenRow, obsWeightsDF) %>%
        as_tibble()

    # if more than one newX was provided, just use first one for weights (for now)
    W <- as.matrix(obsWeightsDF$kernel_weights[[1]])[1,] # may need as.matrix() to convert from sparse to full

    slres = fitPredictLibSL(Y = y_all, X = x_all[, keepX, drop = FALSE], newX = newX[, keepX, drop = FALSE],
                            family = family, SL.library = predict_control$SL.library, verbose = verbose,
                            saveFitLibrary = TRUE, obsWeights = W)
    # will still need to assess fit on newX by using newY and newW

    screenDF = SLScreenExtra:::tidyFeatures.SuperLearner(slres, includeAll = TRUE)
    # this is overkill for one row, but just in case some extra rows slipped in...
    screenDF = screenDF %>%
        group_by(screener) %>%
        do(keep2 = setNames(.$selected, .$term)) %>%
        ungroup()

    predDF = with(slres, tibble(library.predict = library.predict[1,],
                                libraryNames = libraryNames,
                                fitLibrary = unlist(fitLibrary, recursive = FALSE),
                                screener = SL.library$screenAlgorithm,
                                predictor = SL.library$predAlgorithm))

    predres = full_join(predDF, screenDF, by = "screener") %>%
        bind_cols(obsWeightsDF, .)
    # will want to add on 'obsNum' column outside of this function

    return(predres)
}

#' Use a single set of tuning parameters to predict outcomes in the holdout set
#' 
#' @param newX A \code{data.frame} of predictor values for the entire holdout
#' set.
#' @param y_all A numeric vector of outcome values for the entire tuning set.
#' @param x_all A \code{data.frame} of predictor values for the entire tuning
#' set.
#' @param family A character string or function describing the error
#' distribution. Should be one of "gaussian" or "binomial".
#' @param predSummaryRow A single-row \code{data.frame} containing at least
#' columns "combo_method", "selector", "k", "norm", "kernel", "window",
#' "predictor", "screener"
#' @param libraryCVSLFeatSel A list of character vectors of length 2, each
#' containing a screener algorithm and a prediction algorithm, to be used in
#' global feature selection by \code{\link[SuperSelector]{cvSLFeatureSelector}}.
#' See \code{\link[SuperLearner]{SuperLearner}}.
#' @param nCVSLFeatSelFolds Numeric vector of length 1 or 2. Used to specify
#' the number of cross-validation folds for global feature selection in
#' \code{\link[SuperSelector]{cvSLFeatureSelector}}.
#' @param verbose A boolean indicating whether diagnostic messages should be
#' printed. Defaults to \code{FALSE}.
#' @return A named list with elements "predict" (a \code{list}) and "screen" (a
#' \code{data.frame}.
#' @importFrom tibble as_tibble
#' @importFrom dplyr bind_rows
#' @importFrom magrittr `%>%`
#' @importFrom future.apply future_lapply
#' @importFrom SuperSelector cvSLFeatureSelector
#' @export
predictAllYALL <- function(newX, y_all, x_all, family, predSummaryRow, libraryCVSLFeatSel,
                           nCVSLFeatSelFolds, verbose = FALSE) {

    if(length(predSummaryRow[["combo_method"]]) > 1) {
        stop("Only one method")
    }
    if(nrow(predSummaryRow[,c("selector", "k"), drop = FALSE]) > 1) {
        stop("Only one selector")
    }

    cvSLres <- cvSLFeatureSelector(y_all, x_all, family = family,
                                   method = paste("method", predSummaryRow[["combo_method"]], sep = "."),
                                   SL.library = libraryCVSLFeatSel,
                                   selector.library = predSummaryRow[,c("selector", "k"), drop = FALSE],
                                   nFolds = nCVSLFeatSelFolds,
                                   verbose = verbose)

    newXlist = split(newX, 1:nrow(newX))
    # TODO: add warning and alternative if there are no rownames
    names(newXlist) = as.character(rownames(newX))
    predres <- future.apply::future_lapply(newXlist,
                                           predictOneYALL,
                                           y_all = y_all,
                                           x_all = x_all,
                                           family = family,
                                           neighbor_control = list(library = predSummaryRow[,c("norm", "kernel", "window"), drop = FALSE]),
                                           predict_control = list(SL.library = list(as.character(predSummaryRow[,c("predictor", "screener")]))),
                                           keepX = cvSLres$whichVariable$keep_names[[1]],
                                           screenRow = cvSLres$whichVariable[1,],
                                           verbose = verbose)

    predres <- predres %>%
        bind_rows(.id = "obsNum")

    # predres$predict$library.predict - newY
    return(list(predict = predres, screen = as_tibble(cvSLres$summary)))
}

# summarizeYALL_old <- function(df) {
#     # summarize over observations
#     full_join(df %>%
#             # group_by(keep_bin, norm, kernel, window, libraryNames) %>%
#             group_by(norm, kernel, window, libraryNames) %>%
#             dplyr::summarize(cvRiskSum = sum(cvRisk)) %>%
#             arrange(cvRiskSum) %>%
#             ungroup(),
#         df %>%
#             # select(keep_bin, norm, kernel, window, libraryNames, screener, predictor, combo_method, selector, k) %>%
#             select(norm, kernel, window, libraryNames, screener, predictor, combo_method, selector, k) %>%
#             distinct(),
#         # by = c("keep_bin", "norm", "kernel", "window", "libraryNames")) %>%
#         by = c("norm", "kernel", "window", "libraryNames")) %>%
#     select(-libraryNames)
# }

#' Summarize "predict" element of YALL model results
#' 
#' @param df The \code{data.frame} stored in the "predict" element of results
#' from a YALL model
#' @return A \code{data.frame} summary
#' @importFrom dplyr group_by summarize ungroup arrange select
#' @importFrom magrittr `%>%`
#' @export
summarizeYALL <- function(df) {
    # summarize over observations
    df %>%
        group_by(norm, kernel, window, libraryNames, screener, predictor, combo_method, selector, k) %>%
        dplyr::summarize(cvRiskSum = sum(cvRisk)) %>%
        ungroup() %>%
        arrange(cvRiskSum) %>%
        select(-libraryNames)
}

# #' Short description
# #' 
# #' @param df
# #' @return
# #' @importFrom dplyr bind_cols group_by summarize ungroup slice_min select
# #' @importFrom magrittr `%>%`
# bestYALL <- function(df) {
#     bind_cols(df %>%
#         group_by(combo_method, selector, k) %>%
#         dplyr::summarize(cvRiskSum = sum(cvRisk)) %>%
#         ungroup() %>%
#         slice_min(cvRiskSum, with_ties = FALSE) %>%
#         select(-cvRiskSum),
#         df %>%
#             group_by(norm, kernel, window, libraryNames, screener, predictor) %>%
#             dplyr::summarize(cvRiskSum = sum(cvRisk)) %>%
#             ungroup() %>%
#             slice_min(cvRiskSum, with_ties = FALSE) %>%
#             select(-cvRiskSum))
# }

#' Prediction function used by \code{GridLearner}
#' 
#' @param index Index of prediction algorithm to use in \code{lib}
#' @param lib Prediction algorithm (output of
#' \code{SuperLearner:::.createLibrary}).
#' @param Y A numeric vector of outcome values for only the training set.
#' @param dataX A \code{data.frame} of selected predictor values for only the
#' training set.
#' @param newX A single-row \code{data.frame} of selected predictor values for
#' only the current validation observation.
#' @param whichScreen A boolean vector indicating which columns in \code{dataX}
#' and \code{newX} to use in the prediction. 
#' @param family A character string or function describing the error
#' distribution. Should be one of "gaussian" or "binomial".
#' @param id Optional cluster identification variable.
#' @param obsWeights A sparse vector of distances for only the current
#' validation observation with length equal to the number of rows in
#' \code{dataX}.
#' @param verbose A boolean indicating whether diagnostic messages should be
#' printed. Defaults to \code{FALSE}.
#' @param control A list containing two named elements: "saveFitLibrary" (a
#' boolean indicating whether to retain the fit object for each algorithm in the
#' library in the returned result) and "trimLogit" (for the NNloglik combination
#' method, the level to which to truncate the logit transformation). See
#' \code{\link[SuperLearner]{SuperLearner.control}} for details.
#' @param libraryNames A character string label describing the screening and
#' prediction function combination. 
#' @param env Environment containing the prediction functions.
#' @param fitLibEnv Environment containing fitLibrary.
#' @return A numeric vector of predicted outcome values.
the_prediction_function <- function(index, lib, Y, dataX, newX, whichScreen,
    family, id, obsWeights, verbose, control, libraryNames, env, fitLibEnv) {

    pred_fn = get(lib$predAlgorithm[index], envir = env)

    # NOTE: this try(do.call()) is time-consuming
    testAlg <- try(do.call(pred_fn, list(Y = Y,
        X = subset(dataX, select = whichScreen[lib$rowScreen[index], ], drop=FALSE),
        newX = subset(newX, select = whichScreen[lib$rowScreen[index], ], drop=FALSE),
        family = family, id = id, obsWeights = obsWeights)))
    if(inherits(testAlg, "try-error")) {
        warning(paste("Error in algorithm", lib$predAlgorithm[index],
            " on full data", "\n  The Algorithm will be removed from the Super Learner (i.e. given weight 0) \n" ))
        out <- rep.int(NA, times = nrow(newX))
    } else {
        out <- testAlg$pred
        if(control$saveFitLibrary) {
            eval(bquote(fitLibrary[[.(index)]] <- .(testAlg$fit)), envir = fitLibEnv)
        }
    }
    if(verbose) {
        message(paste("full", libraryNames[index]))
    }
    invisible(out)
}

#' De-Supered SuperLearner
#' 
#' The original \code{\link[SuperLearner]{SuperLearner}} was created by Eric
#' Polley in 2011. This function is a modified for purpose version of
#' \code{\link[SuperLearner]{SuperLearner}} which removes the cross-validation
#'  and simply returns the predictions from each library algorithm.
#'
#' @param Y A numeric vector of outcome values for only the training set.
#' @param X A \code{data.frame} of selected predictor values for only the
#' training set.
#' @param newX A single-row \code{data.frame} of selected predictor values for
#' only the current validation observation.
#' @param family A character string or function describing the error
#' distribution. Should be one of "gaussian" or "binomial".
#' @param SL.library A list containing character vectors where the first element
#' is a screening algorithm and the second is a prediction algorithm. See
#' \code{\link[SuperLearner]{SuperLearner}}'s \code{SL.library} argument for
#' more details.
#' @param id Optional cluster identification variable.
#' @param verbose A boolean indicating whether diagnostic messages should be
#' printed. Defaults to \code{FALSE}.
#' @param control A list containing two named elements: "saveFitLibrary" (a
#' boolean indicating whether to retain the fit object for each algorithm in the
#' library in the returned result) and "trimLogit" (for the NNloglik combination
#' method, the level to which to truncate the logit transformation). See
#' \code{\link[SuperLearner]{SuperLearner.control}} for details.
#' @param obsWeights A sparse vector of distances for only the current
#' validation observation with length equal to the number of rows in \code{X}.
#' @param env Environment containing the prediction functions.
#' @return A \code{list} of class \code{SuperLearner}.
#' @import SuperLearner
#' @importFrom stats setNames
#' @export
GridLearner <- function (Y, X, newX = NULL, family = gaussian(), SL.library,
    id = NULL, verbose = FALSE, control = list(), obsWeights = NULL, env = parent.frame()) {

    # get defaults for controls and make sure in correct format
    control <- do.call('SuperLearner.control', control)

    # put together the library
    # should this be in a new environment?
    library <- SuperLearner:::.createLibrary(SL.library)
    SuperLearner:::.check.SL.library(library = c(unique(library$library$predAlgorithm),
                                                        library$screenAlgorithm))

    call <- match.call(expand.dots = TRUE)
    # should we be checking X and newX for data.frame?
    # data.frame not required, but most of the built-in wrappers assume a data.frame
    if(!inherits(X, 'data.frame')) message("X is not a data frame. Check the algorithms in SL.library ",
                                           "to make sure they are compatible with non data.frame inputs")
    varNames <- colnames(X)
    N <- dim(X)[1L]
    p <- dim(X)[2L]
    k <- nrow(library$library)
    kScreen <- length(library$screenAlgorithm)
    Z <- matrix(NA, N, k)
    libraryNames <- paste(library$library$predAlgorithm,
                          library$screenAlgorithm[library$library$rowScreen], sep="_")

    # put fitLibrary in its own environment to locate later
    fitLibEnv <- new.env()
    assign('fitLibrary', vector('list', length = k), envir = fitLibEnv)
    assign('libraryNames', libraryNames, envir = fitLibEnv)
    evalq(names(fitLibrary) <- libraryNames, envir = fitLibEnv)

    # errors* records if an algorithm stops in full data
    errorsInLibrary <- rep(0, k)

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
        family <- get(family, mode="function", envir=parent.frame())
    if(is.function(family))
        family <- family()
    if (is.null(family$family)) {
        print(family)
        stop("'family' not recognized")
    }

    # test id
    if(is.null(id)) {
        id <- seq(N)
    }
    if(!identical(length(id), N)) {
        stop("id vector must have the same dimension as Y")
    }
    # test observation weights
    if(is.null(obsWeights)) {
        obsWeights <- rep(1, N)
    }
    if(!identical(length(obsWeights), N)) {
        stop("obsWeights vector must have the same dimension as Y")
    }

    # now fit all algorithms in library on entire learning data set and predict on newX
    m <- dim(newX)[1L]
    predY <- matrix(NA, nrow = m, ncol = k)
    .screenFun <- function(fun, list) {
        screen_fn = get(fun, envir = env)
        testScreen <- try(do.call(screen_fn, list))
        if (inherits(testScreen, "try-error")) {
            warning(paste("replacing failed screening algorithm,",
                fun, ", with All() in full data", "\n "))
            out <- rep(TRUE, ncol(list$X))
        }
        else {
            out <- testScreen
        }
        return(out)
    }

    whichScreen <- do.call(rbind, sapply(library$screenAlgorithm, FUN = .screenFun,
                                         list = list(Y = Y, X = X, family = family, id = id,
                                                     obsWeights = obsWeights),
                                         simplify = FALSE))

    if(verbose) {
        whichScreen.prnt = whichScreen
        if(!all(colnames(whichScreen.prnt)==varNames)) {
            colnames(whichScreen.prnt) = varNames
        }
        whichScreen.prnt = apply(whichScreen.prnt, 1, function(x) names(which(x)))
        if("All"%in%names(whichScreen.prnt)) {
            whichScreen.prnt = whichScreen.prnt[-which(names(whichScreen.prnt)%in%"All")]
        }
    }

    predY <- do.call('cbind', lapply(seq(k), FUN = the_prediction_function,
        lib = library$library, Y = Y, dataX = X, newX = newX,
        whichScreen = whichScreen, family = family, id = id,
        obsWeights = obsWeights, verbose = verbose, control = control,
        libraryNames = libraryNames, env = env, fitLibEnv = fitLibEnv))

    # check for errors
    errorsInLibrary <- apply(predY, 2, function(xx) any(is.na(xx)))

    # add names of algorithms to the predictions
    colnames(predY) <- libraryNames

    # cvRisk <- apply(Z, 2, function(x) mean(obsWeights * (x - Y)^2))

    out <- list(call = call, libraryNames = libraryNames, SL.library = library,
                coef = setNames(rep(NA, k), libraryNames), library.predict = predY,
                Z = matrix(NA, 1, k), cvRisk = NULL, family = family,
                fitLibrary = get("fitLibrary", envir = fitLibEnv),
                varNames = varNames, whichScreen = whichScreen)
    class(out) <- c("SuperLearner")
    return(out)
}
