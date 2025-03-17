import os
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi import FastAPI, HTTPException
from datetime import datetime
from typing import List
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from catboost import CatBoostClassifier, Pool

Base = declarative_base()
SALT = 'random_salt_for_a_b'
a_percent = 50

class Post(Base):
    __tablename__ = 'post'
    id = Column(Integer, primary_key=True)
    text = Column(String)
    topic = Column(String)


def get_db():
    with SessionLocal() as db:
        return db


SQLALCHEMY_DATABASE_URL = "postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class PostGet(BaseModel):
    id: int
    text: str
    topic: str

    class Config:
        orm_mode = True


class Response(BaseModel):
    exp_group: str
    recommendations: List[PostGet]


def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)

users_control = batch_load_sql('SELECT * FROM t_prokhorenko_user_features_lesson_22')
users_DL = batch_load_sql('SELECT * FROM t_prokhorenko_user_features_lesson_22_dl')

posts_control = batch_load_sql('SELECT * FROM t_prokhorenko_post_features_lesson_22')
posts_DL = batch_load_sql('SELECT * FROM t_prokhorenko_post_features_lesson_22_dl')


def get_model_path(path: str, version: str = 'control') -> str:
    if os.environ.get("IS_LMS") == "1":
        MODEL_PATH = f'/workdir/user_input/model_{version}'
    else:
        MODEL_PATH = path
    return MODEL_PATH



def load_models():
    model_path = get_model_path('model')
    model_control = CatBoostClassifier()
    model_control.load_model(model_path)

    model_path = get_model_path('model', version='test')
    model_test = CatBoostClassifier()
    model_test.load_model(model_path)

    return model_control, model_test


model_control, model_test = load_models()

app = FastAPI()

post_bd = get_db().query(Post).all()


def get_exp_group(user_id: int) -> str:
    import hashlib
    hash = int(hashlib.md5((str(user_id) + SALT).encode()).hexdigest(), 16)
    if  (hash % 100) < a_percent:
        return 'control'
    else:
        return 'test'

@app.get("/post/recommendations", response_model=Response)
async def top_5_recommendations(id: int, time: datetime, limit: int = 10) -> Response:
    if limit <= 0:
        raise HTTPException(status_code=400, detail="Параметрs limit должен быть положительным числом")

    if get_exp_group(id) == 'control':
        users = users_control.copy()
        posts = posts_control.copy()
    else:
        users = users_DL.copy()
        posts = posts_DL.copy()

    user = users[users['user_id'] == id].head(1)
    if user.empty:
        raise HTTPException(status_code=404, detail="Пользователь не найден")
    user.insert(loc=2, column='timestamp', value=time.timestamp())
    exp_group = get_exp_group(id)
    already_liked = user.pop('liked_posts').to_list()

    df = user.merge(posts.loc[~(posts['post_id'].isin(already_liked))], how='cross')
    # cols = df.columns.to_list()
    # cols.remove('post_id')
    # cols.insert(1, 'post_id')
    # df = df[cols]
    X_val = df.drop(columns=['user_id', 'post_id'])
    cat_features = X_val.select_dtypes(include=object).columns.to_list()

    pool = Pool(data=X_val, cat_features=cat_features)
    if exp_group == 'control':
        predict = model_control.predict_proba(pool)[:, 1]
    else:
        predict = model_test.predict_proba(pool)[:, 1]
    df['predict'] = predict

    rec_ids = df.sort_values(by='predict', ascending=False).head(limit).post_id.to_list()
    print(rec_ids)
    recommended_posts = [PostGet.from_orm(post) for post in post_bd if post.id in rec_ids]
    if not recommended_posts:
        return Response(exp_group=exp_group, recommendations=[])
    return Response(exp_group=exp_group, recommendations=recommended_posts)
