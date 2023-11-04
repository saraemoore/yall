#' A wrapper for the mean via \code{glm}.
#' 
#' @param Y Numeric vector containing observed outcome values for the training
#' set.
#' @param X A \code{data.frame} containing predictor values for the training
#' set.
#' @param newX A \code{data.frame} containing predictor values for the test set.
#' @param family A character string or function describing the error
#' distribution. Should be one of "gaussian" or "binomial". 
#' @param obsWeights Optional numeric vector of weights with length equal to
#' the length of \code{Y} and the number of rows in \code{X}.
#' @param ... Currently unused.
#' @return A numeric vector of predictions.
#' @importFrom stats glm predict
#' @export
SL.glm.mean <- function (Y, X, newX, family, obsWeights, ...) {
    fit.glm <- glm(Y ~ 1, data = X, family = family, weights = obsWeights)
    pred <- predict(fit.glm, newdata = newX, type = "response")
    fit <- list(object = fit.glm)
    class(fit) <- "SL.glm"
    out <- list(pred = pred, fit = fit)
    return(out)
}

# SL.glm.interaction.quad = function (Y, X, newX, family, obsWeights, ...) {
#     f <- as.formula(paste0("Y ~ ",
#         paste(colnames(X), collapse= "+"), " + ",
#         "(", paste(colnames(X), collapse= "+"), ")^2 + ",
#         paste("I(", colnames(X), "^2)", collapse= "+")
#     ))

#     fit.glm <- glm(f, family = family, data = X, weights = obsWeights)
#     pred <- predict(fit.glm, newdata = newX, type = "response")
#     fit <- list(object = fit.glm)
#     class(fit) <- "SL.glm"
#     out <- list(pred = pred, fit = fit)
#     return(out)
# }