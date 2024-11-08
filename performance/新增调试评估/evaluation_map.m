function evaluation_map(title,PLS, QLS, GTLS, location)
    

    % Thresholding for binary classification
    threshold = 0.00001; % You may adjust this threshold based on your needs
    PLS_binary = PLS > threshold;
    QLS_filtered_binary = QLS > threshold;

    % Ground Truth Binary
    GTLS_binary = GTLS > 0; % Assuming ground truth values are not probabilities but binary

    % Evaluation Metrics
    TP_PLS = sum(PLS_binary(:) & GTLS_binary(:)); % True Positives for Prior Model
    TP_QLS = sum(QLS_filtered_binary(:) & GTLS_binary(:)); % True Positives for New Model

    FP_PLS = sum(PLS_binary(:) & ~GTLS_binary(:)); % False Positives for Prior Model
    FP_QLS = sum(QLS_filtered_binary(:) & ~GTLS_binary(:)); % False Positives for New Model

    FN_PLS = sum(~PLS_binary(:) & GTLS_binary(:)); % False Negatives for Prior Model
    FN_QLS = sum(~QLS_filtered_binary(:) & GTLS_binary(:)); % False Negatives for New Model

    % Precision, Recall, and F1 Score
    precision_PLS = TP_PLS / (TP_PLS + FP_PLS);
    precision_QLS = TP_QLS / (TP_QLS + FP_QLS);

    recall_PLS = TP_PLS / (TP_PLS + FN_PLS);
    recall_QLS = TP_QLS / (TP_QLS + FN_QLS);

    f1_PLS = 2 * (precision_PLS * recall_PLS) / (precision_PLS + recall_PLS);
    f1_QLS = 2 * (precision_QLS * recall_QLS) / (precision_QLS + recall_QLS);

    % Display or save results
    disp(join([title,' Evaluation Metrics:']));
    disp(['Prior Model Precision: ', num2str(precision_PLS)]);
    disp(['Prior Model Recall: ', num2str(recall_PLS)]);
    disp(['Prior Model F1 Score: ', num2str(f1_PLS)]);
    disp(['New Model Precision: ', num2str(precision_QLS)]);
    disp(['New Model Recall: ', num2str(recall_QLS)]);
    disp(['New Model F1 Score: ', num2str(f1_QLS)]);

    % Optionally, save the results
    results = struct('Prior_Model_Precision', precision_PLS, 'Prior_Model_Recall', recall_PLS, ...
                     'Prior_Model_F1_Score', f1_PLS, 'New_Model_Precision', precision_QLS, ...
                     'New_Model_Recall', recall_QLS, 'New_Model_F1_Score', f1_QLS);
    save(join([location, title,'evaluation_results.mat']), 'results');
end