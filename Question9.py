from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 加载 MNIST 数据集
mnist = fetch_openml('mnist_784', version=1, data_home='./Data')

# 将数据和标签转换为 NumPy 数组
X = mnist.data.to_numpy()
y = mnist.target.to_numpy().astype(int)

# 第一步：拆分出训练集和临时集（70% 训练集，30% 临时集）
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

# 第二步：将临时集拆分为验证集和测试集（20% 验证集，10% 测试集）
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42)

# --------- SVM 训练和评估 ---------
# 初始化并训练 SVM 模型
svc = SVC(kernel='rbf', C=1.0, gamma='scale')
svc.fit(X_train, y_train)

# 使用测试集进行 SVM 预测
y_pred_svm = svc.predict(X_test)

# 计算 SVM 的混淆矩阵
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)

# 绘制 SVM 混淆矩阵
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_svm, annot=True, fmt='d', cmap='Blues')
plt.title('SVM Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# 计算 SVM 的评估指标
accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm, average='weighted')
recall_svm = recall_score(y_test, y_pred_svm, average='weighted')
f1_svm = f1_score(y_test, y_pred_svm, average='weighted')

# 输出 SVM 的评估结果
print(f'SVM Accuracy: {accuracy_svm:.2f}')
print(f'SVM Precision: {precision_svm:.2f}')
print(f'SVM Recall: {recall_svm:.2f}')
print(f'SVM F1-Score: {f1_svm:.2f}')

# --------- 随机森林 训练和评估 ---------
# 初始化随机森林分类器
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)

# 在训练集上训练随机森林模型
rf_clf.fit(X_train, y_train)

# 使用测试集进行随机森林预测
y_pred_rf = rf_clf.predict(X_test)

# 计算随机森林的混淆矩阵
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

# 绘制随机森林混淆矩阵
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues')
plt.title('Random Forest Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# 计算随机森林的评估指标
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf, average='weighted')
recall_rf = recall_score(y_test, y_pred_rf, average='weighted')
f1_rf = f1_score(y_test, y_pred_rf, average='weighted')

# 输出随机森林的评估结果
print(f'Random Forest Accuracy: {accuracy_rf:.2f}')
print(f'Random Forest Precision: {precision_rf:.2f}')
print(f'Random Forest Recall: {recall_rf:.2f}')
print(f'Random Forest F1-Score: {f1_rf:.2f}')

# --------- 对比 SVM 和 随机森林 评估指标 ---------
print("\nComparison of SVM and Random Forest:")
print(f"SVM Accuracy: {accuracy_svm:.2f}, Random Forest Accuracy: {accuracy_rf:.2f}")
print(f"SVM Precision: {precision_svm:.2f}, Random Forest Precision: {precision_rf:.2f}")
print(f"SVM Recall: {recall_svm:.2f}, Random Forest Recall: {recall_rf:.2f}")
print(f"SVM F1-Score: {f1_svm:.2f}, Random Forest F1-Score: {f1_rf:.2f}")
