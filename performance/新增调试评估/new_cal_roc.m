function new_cal_roc(titles,PLS, QLS, GTLS, location)
  
    %disp(sum(GTLS(:)));
    % Thresholding for binary classification
    threshold = 0.1; % You may adjust this threshold based on your needs
    PLS_binary = PLS;%> threshold;
    QLS_binary = QLS;%> threshold;

    % Ground Truth Binary
    GTLS_binary = GTLS > 0.0001; % Assuming ground truth values are not probabilities but binary
    
    PLS_binary = double(PLS_binary);
    QLS_binary = double(QLS_binary);


    % Compute True Positive Rate (TPR) and False Positive Rate (FPR)
    [fpr_PLS, tpr_PLS, ~, auc_PLS] = perfcurve(GTLS_binary(:), PLS_binary(:), true);
    [fpr_QLS, tpr_QLS, ~, auc_QLS] = perfcurve(GTLS_binary(:), QLS_binary(:), true);

    % Plot ROC Curve
    figure;
    plot(fpr_PLS, tpr_PLS, 'b-', 'LineWidth', 2);
    hold on;
    plot(fpr_QLS, tpr_QLS, 'r-', 'LineWidth', 2);
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    title('ROC Curve');
    legend(['Prior Model (AUC = ', num2str(auc_PLS), ')'], ['New Model (AUC = ', num2str(auc_QLS), ')'], 'Location', 'southeast');
    grid on;
    hold off;

    % Display or save results
    disp('Evaluation Metrics:');
    disp(['Prior Model AUC: ', num2str(auc_PLS)]);
    disp(['New Model AUC: ', num2str(auc_QLS)]);

    % Optionally, save the ROC curve
    saveas(gcf, join([location, titles,'roc_curve.png']));
end