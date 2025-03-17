import pandas as pd
from sqlalchemy import create_engine
import nltk
import re
import string
import numpy as np
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV, GroupKFold, train_test_split

engine = create_engine(
    'postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml')
print(1)
pd.set_option('display.max_columns', None)

user_data = pd.read_sql(
    """ SELECT * FROM user_data""",
    con=engine
)
print(2)
post_text_df = pd.read_sql(
    """ SELECT * FROM post_text_df""",
    con=engine
)
print(3)
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
print(4)
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
print('sql complete')

nltk.download('wordnet')
nltk.download('omw-1.4')

wnl = WordNetLemmatizer()


def preprocessing(line, token=wnl):
    line = line.lower()
    line = re.sub(r'[{}]'.format(string.punctuation), " ", line)
    line = line.replace('\n\n', ' ').replace('\n', ' ')
    line = ' '.join([token.lemmatize(x) for x in line.split(' ') if x])
    return line


post_text_df['text_length'] = post_text_df['text'].apply(len)
post_text_df['expression'] = post_text_df['text'].apply(lambda x: x.count('!'))
post_text_df['question'] = post_text_df['text'].apply(lambda x: x.count('?'))
post_text_df['tags'] = post_text_df['text'].apply(lambda x: x.count('#'))
post_text_df['links'] = post_text_df['text'].apply(lambda x: x.count('http'))
post_text_df['len_sen'] = post_text_df['text'].apply(
    lambda x: np.mean([len(i) for i in x.split('.') if i]) if '.' in x else len(x)
)

tfidf = TfidfVectorizer(max_features=10000, stop_words='english', preprocessor=preprocessing)
tfidf_matrix = tfidf.fit_transform(post_text_df['text']).toarray()
tfidf_df = pd.DataFrame(tfidf_matrix, columns=tfidf.get_feature_names_out(), index=post_text_df.index)

post_text_df['TotalTfIdf'] = tfidf_df.sum(axis=1)
post_text_df['MaxTfIdf'] = tfidf_df.max(axis=1)
post_text_df['MeanTfIdf'] = tfidf_df.mean(axis=1)

tfidf_df_centered = tfidf_df.subtract(tfidf_df.mean())
pca = PCA(n_components=110)
pca_result = pca.fit_transform(tfidf_df_centered)
pca_df = pd.DataFrame(data=pca_result, columns=[f'PCA_{i + 1}' for i in range(pca.n_components_)],
                      index=post_text_df.index)

post_text_df = pd.concat([post_text_df.drop('text', axis=1), pca_df], axis=1)
post_text_df = post_text_df.fillna(0)

kmeans = KMeans(n_clusters=15, random_state=0).fit(post_text_df.drop(columns=['post_id', 'topic']))
post_text_df['TextCluster'] = kmeans.labels_

dist_columns = [f"Distance_To_{ith}th_Cluster" for ith in range(1, 16)]
dists_df = pd.DataFrame(
    data=kmeans.transform(post_text_df.drop(columns=['post_id', 'topic', 'TextCluster'])),
    columns=dist_columns,
    index=post_text_df.index
)

post_text_df = pd.concat([post_text_df, dists_df], axis=1)

scaler = StandardScaler()

user_for_scaling = ['gender', 'age', 'exp_group']
user_data[user_for_scaling] = scaler.fit_transform(user_data[user_for_scaling].astype(float))

scaler = StandardScaler()
post_for_scale = post_text_df.drop(columns=['post_id', 'topic']).columns
post_text_df[post_for_scale] = scaler.fit_transform(post_text_df[post_for_scale])

df_ = pd.merge(feed_action, user_data, 'left', 'user_id')
df = pd.merge(df_, post_text_df, 'left', 'post_id')

scaler = StandardScaler()
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S').apply(lambda x: x.timestamp())
df.timestamp = scaler.fit_transform(df['timestamp'].values.reshape(-1, 1))
print('preprocessing complete')
X = df.drop(columns=['user_id', 'post_id', 'target'])
y = df['target']
groups = df['user_id']

class_weight = pd.Series(1 / y.value_counts(normalize=True)).to_list()

X_train, X_test, y_train, y_test, train_groups, test_groups = train_test_split(
    X, y, groups, test_size=0.2, shuffle=False
)

gkf = GroupKFold(n_splits=4)

cat = CatBoostClassifier(random_seed=42, auto_class_weights='Balanced')

object_columns = [
    'topic', 'country', 'city', 'os',
    'source'
]

# Сетка параметров
param_grid = {
    'depth': [2],
    'iterations': [50],
    'learning_rate': [0.3],
    'random_seed': [100],
    'l2_leaf_reg': [70000],
    'cat_features': [object_columns]
}

search = GridSearchCV(cat, param_grid, cv=gkf.split(X_train, y_train, train_groups), scoring='roc_auc', verbose=1)
search.fit(X_train, y_train)
print(search.best_score_)
model = search.best_estimator_

model.save_model('catboost_model',
                 format="cbm")

user_features = user_data.merge(user_likes, on='user_id', how='left')
post_features = post_text_df.copy()

engine = create_engine(
    'postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml')
user_features.to_sql('t_prokhorenko_user_features_lesson_22', con=engine, index=False, if_exists='replace')
post_features.to_sql('t_prokhorenko_post_features_lesson_22', con=engine, index=False, if_exists='replace')
