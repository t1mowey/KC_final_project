import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi import FastAPI, Depends, HTTPException
from datetime import datetime
from typing import List
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from catboost import CatBoostClassifier, Pool

Base = declarative_base()


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


# Модель Post
class PostGet(BaseModel):
    id: int
    text: str
    topic: str

    class Config:
        orm_mode = True


def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)


users = batch_load_sql('SELECT * FROM t_prokhorenko_user_features_lesson_22')
posts = batch_load_sql('SELECT * FROM t_prokhorenko_post_features_lesson_22')
import os


def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH


def load_models():
    model_path = get_model_path('catboost_model')
    from_file = CatBoostClassifier()
    from_file.load_model(model_path)
    return from_file


loaded_model = load_models()

app = FastAPI()

post_bd = get_db().query(Post).all()

@app.get("/post/recommendations", response_model=List[PostGet])
async def top_5_recommendations(id: int, time: datetime, limit: int = 10, db=Depends(get_db)) -> List[PostGet]:
    if limit <= 0:
        raise HTTPException(status_code=400, detail="Параметрs limit должен быть положительным числом")

    user = users[users['user_id'] == id].head(1)
    already_liked = list(user['liked_posts'])
    user.drop(columns='liked_posts', inplace=True)

    if user.empty:
        raise HTTPException(status_code=404, detail="Пользователь не найден")
    user.insert(loc=2, column='timestamp', value=time.timestamp())

    df = user.merge(posts.loc[~(posts['post_id'].isin(already_liked))], how='cross')
    cols = list(df.columns)
    cols.remove('post_id')
    cols.insert(1, 'post_id')
    df = df[cols]
    X_val = df.drop(columns=['user_id', 'post_id'])

    pool = Pool(data=X_val, cat_features=['topic', 'country', 'city', 'os','source'])
    predict = loaded_model.predict_proba(pool)[:, 1]
    df['predict'] = predict
    rec_ids = df.sort_values(by='predict', ascending=False).head(limit).post_id.to_list()
    print(rec_ids)
    recommended_posts = [PostGet.from_orm(post) for post in post_bd if post.id in rec_ids]
    if not recommended_posts:
        raise HTTPException(status_code=200, detail='recommended_posts is empty')
    return recommended_posts
