import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# 读取数据
train_data = pd.read_csv("train_set.csv", sep='\t')
test_data = pd.read_csv("test_a.csv", sep='\t')

# 探索数据
print(train_data.shape)
print(train_data.head())

# 将数据分为训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(train_data['text'], train_data['label'], test_size=0.2, random_state=42)

# TF-IDF 特征提取
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
train_tfidf = tfidf_vectorizer.fit_transform(X_train)
val_tfidf = tfidf_vectorizer.transform(X_val)
test_tfidf = tfidf_vectorizer.transform(test_data['text'])

# 建立岭回归分类器
clf = RidgeClassifier(alpha=1.0, solver='sag', random_state=42)
clf.fit(train_tfidf, y_train)

# 在验证集上评估模型
val_pred = clf.predict(val_tfidf)
val_f1 = f1_score(y_val, val_pred, average='macro')
print(f"验证集上的F1分数: {val_f1}")

# 对测试集进行预测
test_pred = clf.predict(test_tfidf)

# 创建提交结果的DataFrame并保存为CSV文件
submit_df = pd.DataFrame()
submit_df['label'] = test_pred
submit_df.to_csv('submit.csv', index=None)

print("提交文件 'submit.csv' 生成成功。")
