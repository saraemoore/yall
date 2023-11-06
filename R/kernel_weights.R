#' Scale a numeric vector according to a window size
#' 
#' @param d A numeric vector of distances. In a typical use case, \code{d} will
#' have length equal to the number of training set observations.
#' @param windowProp Window size, as a proportion (0, 1]. Defaults to
#' \code{0.20}.
#' @param verbose A boolean indicating whether diagnostic messages should be
#' printed. Defaults to \code{FALSE}.
#' @return A numeric vector with length equal to the length of \code{d}.
#' @importFrom SuperSelector makeLibArgNumeric
#' @importFrom scales percent
scaleDistForKernel <- function(d, windowProp = 0.20, verbose = FALSE) {
    xStop = function(txt = "") {
        stop(paste("Window size must be coercible to numeric.", txt))
    }
    windowProp = makeLibArgNumeric(windowProp, xStop)
    if(!is.numeric(windowProp)|windowProp <= 0|windowProp > 1) {
        stop("Invalid window size supplied. Please supply a valid proportion of observations.")
    }

    refRank = ceiling(length(d)*windowProp)
    # denominator is metric window size, window width, bandwidth, or scale parameter.
    # because we want the neighborhood size to be fixed, this is adaptive.
    if(refRank < sum(d==0)) {
        refRank <- sum(d==0)
        if(verbose) {
            warning("Distances are zero in window. Window automatically expanded from ",
                    percent(windowProp), " to ",
                    percent(refRank/length(d)), ".")
        }
    }

    dOrder = order(d)
    d = d[dOrder]

    refRankRange = c(refRank - 1, refRank)
    if(refRank < length(d)) {
        refRankRange = refRankRange + 1
    }
    if(verbose) {
        message("Window will include observations up to and including observation number ",
            refRank, " (of ", length(d),
            ") by scaling all distances by the distance of observation number ", refRank,
            " plus half the difference between observation numbers ", refRankRange[1],
            " and ", refRankRange[2], ".")
    }
    scaleFactor = d[refRank] + diff(d[refRankRange])/2
    if(diff(d[c(1,refRank)])==0 & scaleFactor==0) {
        # TODO: need better error message
        #   (although this error shouldn't arise now that refRank is checked above)
        # other solutions here could include allowing more covariates to
        #   'survive' global feature selection
        stop("Distances are zero in window. Please specify a larger window size.")
    }
    return((d/scaleFactor)[order(dOrder)])
}

#' Row-wise scaling according to a window size
#' 
#' @param windowProp Window size, as a proportion (0, 1].
#' @param d a matrix with rows corresponding to test set observations and
#' columns corresponding to training set observations
#' @param verbose A boolean indicating whether diagnostic messages should be
#' printed. Defaults to \code{FALSE}.
#' @return a matrix with rows corresponding to test set observations and columns
#' corresponding to training set observations
scaleRowDists <- function(windowProp, d, verbose = FALSE) {
    # for each test set observation, ...
    res = apply(d, 1, scaleDistForKernel, windowProp = windowProp, verbose = verbose)
    # TODO: parallelize
    return(t(res))
}

#' Row-wise application of a kernel function
#' 
#' @param kernelFunction A character string denoting the kernel function to be
#' used. See \code{\link[kedd]{kernel.fun}} for details.
#' @param d a matrix with rows corresponding to test set observations and
#' columns corresponding to training set observations
#' @param verbose A boolean indicating whether diagnostic messages should be
#' printed. Defaults to \code{FALSE}.
#' @return a matrix with rows corresponding to test set observations and
#' columns corresponding to training set observations
weightRowNeighbors = function(kernelFunction, d, verbose = FALSE) {
    res = apply(d, 1, weightNeighbors, kernelFunction = kernelFunction, verbose = verbose)
    # TODO: parallelize
    return(t(res))
}

#' Apply a kernel function to a numeric vector
#' 
#' @param scaledDist A numeric vector
#' @param kernelFunction A character string denoting the kernel function to be
#' used. Defaults to "gaussian". See \code{\link[kedd]{kernel.fun}} for details.
#' @param verbose Currently unused. A boolean indicating whether diagnostic
#' messages should be printed. Defaults to \code{FALSE}.
#' @return A numeric vector with length equal to the length of
#' \code{scaledDist}.
#' @importFrom kedd kernel.fun
weightNeighbors = function(scaledDist, kernelFunction, verbose = FALSE) {
    if(missing(kernelFunction)) {
        kernelFunction = "gaussian"
    }
    kedd::kernel.fun(scaledDist, kernel = kernelFunction)$kx
    # kernelResInv = 1/kernelRes[(kernelRes > 0)]
}

#' Estimate distances in "neighborhoods" using specified features
#' 
#' @param keepX Character vector containing names of retained features.
#' @param trainX A \code{data.frame} (typically containing the training/tuning
#' set) to be transformed.
#' @param testX A \code{data.frame} containing one or more observations to
#' transform \code{trainX} against (typically a validation/test set).
#' @param neighbor_library A \code{data.frame} containing columns "norm",
#' "kernel", and "window", with one row for each unique combination of settings
#' that should be applied. Column "norm" should contain valid norm function
#' names (see the \code{normF} argument of \code{\link{distByRows}} for
#' options). Column "kernel" should contain valid kernel function names (see the
#' \code{kernel} argument of \code{\link[kedd]{kernel.fun}} for options). Column
#' "window" should contain valid window sizes (see the \code{windowProp}
#' argument of \code{\link{scaleRowDists}} for details).
#' @param save_dist A boolean indicating whether to retain the raw distance
#' matrices in the resulting \code{data.frame} in an additional (list) column
#' named "dist". Defaults to \code{FALSE}.
#' @param verbose A boolean indicating whether diagnostic messages should be
#' printed. Defaults to \code{FALSE}.
#' @return The \code{neighbor_library} \code{data.frame} supplemented with at
#' least two columns: "windowFmtd" (a formatted string version of the supplied
#' column "window") and "kernel_weights". Each row's "kernel_weights" contains a
#' sparse matrix of distances with number of rows equal to the number of rows in
#' \code{testX} and the number of and number of columns equal to the number of
#' rows in \code{trainX}.
#' @importFrom dplyr select distinct mutate full_join
#' @importFrom magrittr `%>%`
#' @importFrom purrr map map2
#' @importFrom scales percent
#' @importFrom methods as
#' @export
#' @examples
#' neighbor_lib <- tidyr::expand_grid(norm = c("L1", "L2"),
#'                                    kernel = c("uniform", "tricube", "epanechnikov"),
#'                                    window = c(0.1, 0.2, 0.3))
#' res <- distKernelWeights(keepX = c("qsec", "cyl", "disp", "mpg", "carb", "drat", "hp", "wt"),
#'                          trainX = mtcars[1:28, -which(colnames(mtcars) %in% "vs")],
#'                          testX = mtcars[29:32, -which(colnames(mtcars) %in% "vs")],
#'                          neighbor_library = neighbor_lib)
distKernelWeights = function(keepX, trainX, testX, neighbor_library, save_dist = FALSE, verbose = FALSE) {

    # step over each unique norm
    normCombos = neighbor_library %>%
        dplyr::select(norm) %>%
        dplyr::distinct()

    #### Input 1 must be a vector of column names, not a list.
    d = normCombos %>% dplyr::mutate(dist = purrr::map(norm, distByRows, testX, trainX, keepX, verbose))

    # step over each unique window + norm combo
    # left_join, right_join, or inner_join might be faster here than full_join
    normWindowCombos = neighbor_library %>%
        dplyr::select(norm, window) %>%
        dplyr::distinct() %>%
        dplyr::full_join(d, by = "norm")

    scaledDist = normWindowCombos %>%
        dplyr::mutate(scaled_dist = purrr::map2(window, dist, scaleRowDists, verbose))

    # step over each kernel + window + norm combo
    # left_join, right_join, or inner_join might be faster here than full_join
    kernelNormWindowCombos = neighbor_library %>%
        dplyr::full_join(scaledDist, by = c("norm", "window"))
    kernelNormWindowCombos = kernelNormWindowCombos %>%
        dplyr::mutate(windowFmtd = scales::percent(window))
    kernelRes = kernelNormWindowCombos %>%
        dplyr::mutate(kernel_weights = purrr::map2(kernel, scaled_dist, weightRowNeighbors, verbose))

    kernelRes = kernelRes %>%
        dplyr::mutate(kernel_weights = purrr::map(kernel_weights, as, "sparseMatrix"))
    
    # set the names

    if(!save_dist) {
        kernelRes <- kernelRes %>%
            dplyr::select(-dist, -scaled_dist)
    }
    return(kernelRes)
}
