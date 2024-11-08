% 创建一个复数矩阵
complex_matrix = [1 + 2i, 3 + 4i; 5 + 6i, 7 + 8i];

% 将复数矩阵的虚部置零，转换成双精度类型的实数矩阵
double_matrix = real(complex_matrix);

disp('原始复数矩阵：');
disp(complex_matrix);

disp('转换后的双精度实数矩阵：');
disp(double_matrix);
quantile(prio(:),l);