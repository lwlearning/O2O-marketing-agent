import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from src.config import settings

def generate_o2o_sample_data():
    np.random.seed(42)
    n_samples = 10000
    data = {
        "User_id": np.arange(n_samples),
        "Coupon_id": np.random.choice([1,2,3,4,5,np.nan], n_samples),
        "Discount": np.random.choice(["20:100", "10:50", "9折", "8折", np.nan], n_samples),
        "Distance": np.random.randint(0, 10, n_samples),
        "Purchase": np.random.randint(0, 2, n_samples)
    }
    df = pd.DataFrame(data)
    os.makedirs(settings.KNOWLEDGE_BASE_DIR, exist_ok=True)
    df.to_csv(f"{settings.KNOWLEDGE_BASE_DIR}/o2o_marketing.csv", index=False, encoding="utf-8")
    return df

def build_uplift_dataset(df):
    df["treatment"] = df["Coupon_id"].notna().astype(int)
    df["conversion"] = df["Purchase"]
    return df

def ipw_correction(df):
    X = df[["Distance"]].fillna(0)
    ps_model = LogisticRegression(random_state=42)
    ps_model.fit(X, df["treatment"])
    df["propensity_score"] = ps_model.predict_proba(X)[:,1]
    df["ipw_weight"] = np.where(df["treatment"]==1, 1/df["propensity_score"], 1/(1-df["propensity_score"]))
    return df

def calculate_uplift(df):
    treat = df[df["treatment"]==1]
    control = df[df["treatment"]==0]
    p_treat = np.average(treat["conversion"], weights=treat["ipw_weight"])
    p_control = np.average(control["conversion"], weights=control["ipw_weight"])
    df["uplift_score"] = p_treat - p_control

    def segment(row):
        if row["uplift_score"]>0.1 and row["conversion"]==0:
            return "说服型（必发券）"
        elif row["conversion"]==1:
            return "天然转化（不发券）"
        elif row["uplift_score"] < -0.05:
            return "逆反型（禁发）"
        else:
            return "沉睡型"
    df["user_segment"] = df.apply(segment, axis=1)
    return df

def generate_rag_knowledge(df):
    knowledge = []
    for _, row in df.sample(2000).iterrows():
        text = f"""
用户ID：{row.User_id}
分层：{row.user_segment}
商户距离：{row.Distance}km
干预：{"发券" if row.treatment==1 else "不发券"}
优惠：{row.Discount}
Uplift增量：{row.uplift_score:.2f}
结果：{"转化" if row.conversion==1 else "未转化"}
        """.strip()
        knowledge.append(text)
    with open(f"{settings.KNOWLEDGE_BASE_DIR}/uplift_knowledge.md","w",encoding="utf-8") as f:
        f.write("\n\n---\n\n".join(knowledge))

def run_data_pipeline():
    print("🔵 开始天池O2O数据处理 + Uplift纠偏...")
    df = generate_o2o_sample_data()
    df = build_uplift_dataset(df)
    df = ipw_correction(df)
    df = calculate_uplift(df)
    generate_rag_knowledge(df)
    print("✅ 数据处理完成！")
    return df