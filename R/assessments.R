#' Cross-validated AUC estimate with confidence interval as a formatted string
#' 
#' @param yhat A vector of predicted outcome values
#' @param y A vector of observed outcome values
#' @param folds Vector of fold IDs. If \code{NULL}, the standard error will be
#' estimated as though there were a single fold (no cross-validation).
#' @param confidence Confidence level; defaults to \code{0.95}
#' @return A character string
#' @importFrom cvAUC ci.cvAUC
pretty_cvauc_ci <- function(yhat, y, folds = NULL, confidence = 0.95) {
    res_cvauc <- cvAUC::ci.cvAUC(predictions = yhat, labels = y,
                                 folds = folds, confidence = confidence)
    sprintf("AUC = %.3f (%.3f, %.3f)", res_cvauc$cvAUC, res_cvauc$ci[1], res_cvauc$ci[2])
}

#' Prepare data.frames for multiple model assessment visualizations
#' 
#' @param yhat A vector of predicted outcome values
#' @param y A vector of observed outcome values
#' @param yhat_label Optional label to describe the model output. Useful when
#' multiple models will be compared in a single visualization. Defaults to
#' "Predictions".
#' @return A named list containing 4 \code{data.frame}s: "res" (containing
#' predicted and observed outcome values), "roc" (ROC curve), "pr"
#' (precision-recall curve), and "acc" (accuracy curve).
#' @importFrom yardstick pr_auc_vec
#' @importFrom ROCR prediction performance
#' @export
assess_predictions <- function(yhat, y, yhat_label = "Predictions") {
    auc_res <- pretty_cvauc_ci(yhat, y)

    aupr_res <- yardstick::pr_auc_vec(factor(y, levels = c(1, 0)), yhat, event_level = "first")

    roc_pred <- ROCR::prediction(yhat, y)
    roc_res <- ROCR::performance(roc_pred, measure = "tpr", x.measure = "fpr")
    prec_res <- ROCR::performance(roc_pred, measure = "prec", x.measure = "rec")
    acc_res <- ROCR::performance(roc_pred, measure = "acc")

    roc_auc_df <- data.frame(fpr = roc_res@x.values[[1]], tpr=roc_res@y.values[[1]],
                             auc.details = auc_res)
    prec_rec_df <- data.frame(rec = prec_res@x.values[[1]], prec = prec_res@y.values[[1]],
                              auc.details = aupr_res)
    acc_df <- data.frame(cutoff = acc_res@x.values[[1]], acc = acc_res@y.values[[1]])

    roc_auc_df$method = paste(yhat_label, auc_res, sep = "\n")
    prec_rec_df$method = paste(yhat_label, paste("AUCPR =", round(aupr_res, 3)), sep = "\n")

    return(list(res = data.frame(y = y, yhat = yhat), roc = roc_auc_df, pr = prec_rec_df, acc = acc_df))
}