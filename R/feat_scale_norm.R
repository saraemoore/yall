#' Calculate the magnitude of a vector
#'  
#' @param x a numeric vector
#' @param f a character string representing one of the following functions:
#' * \code{L1}: Manhattan aka 1-norm or L1-norm (default)
#' * \code{L2}: Euclidean aka 2-norm or L2-norm
#' * \code{L3}: Minkowski aka 3-norm or L3-norm
#' * \code{L4}: Minkowski aka 4-norm or L4-norm
#' * \code{L5}: Minkowski aka 5-norm or L5-norm
#' * \code{L6}: Minkowski aka 6-norm or L6-norm
#' * \code{max}: Chebyshev aka infinity norm
#' @return a numeric value
vectorNorm = function(x, f = c("L1", "L2", "L3", "L4", "L5", "L6", "max")) {
    f = match.arg(f)
    switch(as.character(f),
        L1 = sum(abs(x)),
        L2 = sqrt(sum(x^2)),
        L3 = sum(abs(x)^3)^(1/3),
        L4 = sum(x^4)^(1/4),
        L5 = sum(abs(x)^5)^(1/5),
        L6 = sum(x^6)^(1/6),
        max = max(abs(x))
        # TODO: add Mahalanobis distance? might not work
    )
}

#' Calculate the magnitude of row-vectors
#' 
#' @param x A \code{data.frame} or \code{matrix}
#' @param f a character string representing one of the following functions:
#' * \code{L1}: Manhattan aka 1-norm or L1-norm (default)
#' * \code{L2}: Euclidean aka 2-norm or L2-norm
#' * \code{L3}: Minkowski aka 3-norm or L3-norm
#' * \code{L4}: Minkowski aka 4-norm or L4-norm
#' * \code{L5}: Minkowski aka 5-norm or L5-norm
#' * \code{L6}: Minkowski aka 6-norm or L6-norm
#' * \code{max}: Chebyshev aka infinity norm
#' @return a numeric vector
rowNorms = function(x, f = c("L1", "L2", "L3", "L4", "L5", "L6", "max")) {
    f = match.arg(f)
    switch(as.character(f),
        L1 = rowSums(abs(x)),
        L2 = sqrt(rowSums(x^2)),
        L3 = rowSums(abs(x)^3)^(1/3),
        L4 = rowSums(x^4)^(1/4),
        L5 = rowSums(abs(x)^5)^(1/5),
        L6 = rowSums(x^6)^(1/6),
        max = apply(abs(x), 1, max)
        # TODO: add Mahalanobis distance? might not work
    )
    # canberra aka weighted L_1 distance: not useful here b/c centroid is 0 for all features
    # One-minus-Pearson-correlation: also not useful here b/c centroid is 0 for all features
}

#' Normalize a dataset to a single observation
#'
#' Normalize a dataset \code{X} to a single observation \code{obs} by
#' subtracting the observation's values and dividing by the L1 norm.
#' 
#' @param obs A \code{data.frame} containing a single record/row (typically a
#' validation/test set observation) to which \code{X} will be normalized. 
#' @param X A \code{data.frame} (typically containing the training set) to be
#' normalized.
#' @param keepX Character vector containing names of retained features.
#' @return A \code{data.frame}
#' @importFrom dplyr bind_rows select one_of mutate_all slice n
#' @importFrom magrittr `%>%`
normalizeFeatures <- function(obs, X, keepX = colnames(X)) {
    bind_rows(obs, X) %>%
        # only keep columns of X in keepX
        select(one_of(keepX)) %>%
        # center each column by its value in obs
        mutate_all(list(~ . - .[1])) %>%
        # scale each column by its L1 norm.
        # could instead do norm(as.matrix(.), "1"), but sum(abs(.)) is faster.
        # when a column is completely homogeneous, this step produces all NaNs:
        mutate_all(list(~ ./vectorNorm(.))) %>%
        # remove obs that was prepended
        slice(2:n()) %>%
        # tibbles are trouble
        as.data.frame()
}

#' Estimate the record-level distance from \code{validX} to \code{trainX}
#'
#' @param normF a character string representing one of the following functions:
#' * \code{L1}: Manhattan aka 1-norm or L1-norm (default)
#' * \code{L2}: Euclidean aka 2-norm or L2-norm
#' * \code{L3}: Minkowski aka 3-norm or L3-norm
#' * \code{L4}: Minkowski aka 4-norm or L4-norm
#' * \code{L5}: Minkowski aka 5-norm or L5-norm
#' * \code{L6}: Minkowski aka 6-norm or L6-norm
#' * \code{max}: Chebyshev aka infinity norm
#' @param validX A \code{matrix}, \code{data.frame}, or numeric vector
#' containing one or more observations to which \code{trainX} should be
#' normalized (typically a validation/test set).
#' @param trainX A \code{data.frame} (typically containing the training set) to
#' be normalized.
#' @param keepX Character vector containing names of retained features.
#' @param verbose Currently unused. A boolean indicating whether diagnostic
#' messages should be printed. Defaults to \code{FALSE}.
#' @return A \code{matrix}
distByRows <- function(normF = c("L1", "L2", "L3", "L4", "L5", "L6", "max"),
                       validX, trainX, keepX, verbose = FALSE) {
    normF = match.arg(normF)

    if(is.vector(validX)|is.matrix(validX)) {
        if(is.vector(validX)) {
            validX = as.data.frame(t(validX))
        }
        colnames(validX) = colnames(trainX)
    }

    # for every validation set observation:
    newTrainX = apply(validX, 1, normalizeFeatures, trainX, keepX)
    # now we have a list with length = number of rows in validX
    # each element is a new trainX data.frame tailored to that validation set observation

    # sapply(norms, function(d) t(sapply(newTrainX, rowNorms, d)), simplify = FALSE)
    res = sapply(newTrainX, rowNorms, normF)
    # TODO: parallelize
    return(t(res))
}
