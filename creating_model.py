import pandas as pd
import time
from sqlalchemy import create_engine

engine = create_engine(
    'postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml')
pd.set_option('display.max_columns', None)

print(f'start sql-pulling')

user_data = pd.read_sql(
    """ SELECT * FROM user_data""",
    con=engine
)
post_text_df = pd.read_sql(
    """ SELECT * FROM t_prokhorenko_post_features_lesson_10_roberta""",
    con=engine
)

feed_action = pd.read_sql("""SELECT * FROM feed_data WHERE action = 'view' LIMIT 700000""", con=engine)
feed_action.drop(columns='action', inplace=True)

like_probabilities = pd.read_sql("""
    SELECT 
    post_id,
    AVG(target) AS like_probabilities
    FROM feed_data
    GROUP BY 
        post_id
""", con=engine)
post_text_df = post_text_df.merge(like_probabilities, on='post_id', how='left')

user_likes = pd.read_sql("""
    SELECT user_id, ARRAY_AGG(post_id) AS liked_posts 
    FROM feed_data 
    WHERE target = 1
    GROUP BY user_id
""", con=engine)

post_likes = pd.read_sql("""
    SELECT 
    post_id,
    COALESCE(COUNT(CASE WHEN target = 1 THEN 1 END), 0) AS likes_count,
    COALESCE(COUNT(CASE WHEN target = 0 THEN 1 END), 0) AS views_count
FROM 
    feed_data
GROUP BY 
    post_id;
""", con=engine)
post_text_df = post_text_df.merge(post_likes, on='post_id', how='left')

user_like_chance = pd.read_sql("""
    SELECT user_id, AVG(target) AS user_like_chance
    FROM feed_data 
    GROUP BY user_id
""", con=engine)
user_data = user_data.merge(user_like_chance, on='user_id', how='left')

print(f'complete sql-pulling')

embedding_features = [f'emb_{i}' for i in range(768)]
text_embeddings = post_text_df[embedding_features]

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# # Добавление дополнительных текстовых метрик
# post_text_df['text_length'] = post_text_df['text'].apply(len)
# post_text_df['expression'] = post_text_df['text'].apply(lambda x: x.count('!'))
# post_text_df['question'] = post_text_df['text'].apply(lambda x: x.count('?'))
# post_text_df['tags'] = post_text_df['text'].apply(lambda x: x.count('#'))
# post_text_df['links'] = post_text_df['text'].apply(lambda x: x.count('http'))
# post_text_df['len_sen'] = post_text_df['text'].apply(
#     lambda x: np.mean([len(i) for i in x.split('.') if i]) if '.' in x else len(x)
# )

post_text_df['max_emb'] = text_embeddings.max(axis=1)
post_text_df['mean_emb'] = text_embeddings.mean(axis=1)
post_text_df['min_emb'] = text_embeddings.min(axis=1)

# Центрирование данных
text_embeddings_centered = text_embeddings.subtract(text_embeddings.mean())

# Инициализация и применение PCA
pca = PCA(n_components=20)
pca_result = pca.fit_transform(text_embeddings_centered)

# Преобразование результата PCA в DataFrame и сохранение исходных индексов
pca_df = pd.DataFrame(data=pca_result, columns=[f'PCA_{i + 1}' for i in range(pca.n_components_)],
                      index=post_text_df.index)
pca_features = pca_df.columns.to_list()

# Объединение DataFrame
post_text_df = pd.concat([post_text_df.drop(['text'] + text_embeddings.columns.to_list(), axis=1), pca_df],
                         axis=1).fillna(0)

# Кластеризация с использованием KMeans
n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(post_text_df[pca_features])

# Добавление кластеров в DataFrame
post_text_df['TextCluster'] = kmeans.labels_

# Вычисление расстояний до кластеров
dist_columns = [f"Distance_To_{i}th_Cluster" for i in range(1, n_clusters + 1)]
dists_df = pd.DataFrame(
    data=kmeans.transform(post_text_df[pca_features]),
    columns=dist_columns,
    index=post_text_df.index
)

# Объединение расстояний до кластеров с основным DataFrame
post_text_df = pd.concat([post_text_df, dists_df], axis=1)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

user_for_scaling = ['gender', 'age', 'exp_group']
user_data[user_for_scaling] = scaler.fit_transform(user_data[user_for_scaling].astype(float))

scaler = StandardScaler()

post_for_scale = post_text_df.drop(columns=['post_id', 'topic']).columns.to_list()
post_text_df[post_for_scale] = scaler.fit_transform(post_text_df[post_for_scale])

df_ = pd.merge(feed_action, user_data, 'left', 'user_id')
df = pd.merge(df_, post_text_df, 'left', 'post_id')

df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S').apply(lambda x: x.timestamp())
# df.timestamp = scaler.fit_transform(df['timestamp'].values.reshape(-1, 1))

print(f'df is ready, start fitting')

from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV, GroupKFold, train_test_split

X = df.drop(columns=['user_id', 'post_id', 'target'])  # Замените 'target' на имя вашей целевой переменной
y = df['target']
groups = df['user_id']  # Группировка по пользователям

class_weight = pd.Series(1 / y.value_counts(normalize=True)).to_list()

# Шаг 1: Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test, train_groups, test_groups = train_test_split(
    X, y, groups, test_size=0.2, shuffle=False
)

# Создание объекта GroupKFold
gkf = GroupKFold(n_splits=4)

# Инициализация модели
cat = CatBoostClassifier(random_seed=42, auto_class_weights='Balanced')

object_columns = [
    'topic', 'country', 'city', 'os',
    'source'
]

# Сетка параметров
param_grid = {
    'depth': [2],
    'iterations': [100],
    'learning_rate': [0.3],
    'random_seed': [100],
    'l2_leaf_reg': [0],
    'cat_features': [object_columns]
}

# GridSearchCV с использованием GroupKFold
search = GridSearchCV(cat, param_grid, cv=gkf.split(X_train, y_train, train_groups), scoring='roc_auc', verbose=0)

# Шаг 2: Обучение модели
search.fit(X_train, y_train)
print(search.best_params_)
print(search.best_score_)
model = search.best_estimator_

model.save_model('catboost_model',
                 format="cbm")

from_file = CatBoostClassifier()  # здесь не указываем параметры, которые были при обучении, в дампе модели все есть

from_file.load_model("catboost_model")

from_file.predict(X_test)

print(f'model saved, start sql-pushing')

user_features = user_data.merge(user_likes, on='user_id', how='left')

post_features = post_text_df.copy()

user_features.to_sql('t_prokhorenko_user_features_lesson_22', con=engine, index=False,
                     if_exists='replace')  # записываем таблицу
post_features.to_sql('t_prokhorenko_post_features_lesson_22', con=engine, index=False, if_exists='replace')
