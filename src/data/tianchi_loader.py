# src/data/tianchi_loader.py
import os
import pandas as pd
import zipfile
from src.core.config import settings
from src.core.logger import get_logger

logger = get_logger(__name__)

class TianchiDataLoader:
    def __init__(self):
        self.raw_data_path = os.path.join(settings.DATA_RAW_DIR, "tianchi_o2o")
        self.processed_data_path = os.path.join(settings.DATA_PROCESSED_DIR, "user_features.csv")

    def load_raw_data(self, use_online: bool = True) -> pd.DataFrame:
        offline_train_file = os.path.join(self.raw_data_path, "offline_train.csv.zip")
        if not os.path.exists(offline_train_file):
            raise FileNotFoundError(f"请先下载数据集到 {self.raw_data_path}")

        with zipfile.ZipFile(offline_train_file, 'r') as zip_ref:
            file_list = [f for f in zip_ref.namelist() if f.endswith(".csv") and not f.startswith("__MACOSX")]
            csv_file = file_list[0]
            with zip_ref.open(csv_file) as f:
                df_offline = pd.read_csv(f)

        logger.info(f"✅ 成功加载线下数据: {len(df_offline)} 条记录")
        return df_offline

    def _classify_user_static(self, distance_km: float, historical_spend: int, is_new_user: bool):
        # 1. 超远距离一律低价值
        if distance_km > 5:
            return "低价值用户"
        # 2. 真实新用户
        if is_new_user and historical_spend == 0:
            return "平台新用户"
        # 3. 沉睡老用户
        if historical_spend == 0:
            return "沉睡无效用户"
        # 4. 活跃用户按距离分层
        if distance_km <= 2:
            return "近场用户"
        elif 2 < distance_km <= 5:
            return "远场用户"
        return "普通用户"

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        # 距离转换
        def distance_to_km(x):
            if pd.isna(x): return 5.0
            x = float(x)
            if x == 0: return 0.3
            if x == 10: return 6.0
            return x * 0.5

        df["distance_km"] = df["Distance"].apply(distance_to_km)
        df["is_used"] = df["Date"].notna().astype(int)
        df["is_treated"] = df["Coupon_id"].notna().astype(int)

        # 解析折扣
        def parse_discount(x):
            if pd.isna(x): return 0, 0
            s = str(x)
            if ":" in s:
                try:
                    a, b = s.split(":"); return float(a), float(b)
                except:
                    return 0, 0
            return 0, 0

        df[["threshold", "discount"]] = df["Discount_rate"].apply(lambda x: pd.Series(parse_discount(x)))

        # 时间特征
        df["Date_received"] = pd.to_datetime(df["Date_received"], errors="coerce")
        user_first_date = df.groupby("User_id")["Date_received"].min().reset_index()
        user_first_date.rename(columns={"Date_received": "first_receive_date"}, inplace=True)

        # 用户聚合特征
        user_features = df.groupby("User_id").agg(
            historical_spend=("is_used", "sum"),
            total_coupons=("is_treated", "sum"),
            coupon_use_rate=("is_used", "mean"),
            avg_distance=("distance_km", "mean"),
            avg_discount=("discount", "mean")
        ).reset_index()

        # 合并时间特征
        user_features = user_features.merge(user_first_date, on="User_id")
        dataset_last_date = df["Date_received"].max()
        user_features["is_new_user"] = user_features["first_receive_date"] >= (dataset_last_date - pd.Timedelta(days=30))

        # =======================
        # 🔥 预计算所有用户的分群（只算这一次！）
        # =======================
        user_features["user_segment"] = user_features.apply(
            lambda row: self._classify_user_static(
                distance_km=row["avg_distance"],
                historical_spend=row["historical_spend"],
                is_new_user=row["is_new_user"]
            ), axis=1
        )

        return user_features

    def count_user_segments(self):
        """直接读取预计算的分群，不重复计算"""
        logger.info("📊 正在统计全量用户分层数量...")
        if not os.path.exists(self.processed_data_path):
            raw = self.load_raw_data()
            feat = self.preprocess(raw)
            self.save_processed_data(feat)
        else:
            feat = pd.read_csv(self.processed_data_path)

        segment_count = feat["user_segment"].value_counts()
        total_users = len(feat)

        logger.info("=" * 50)
        logger.info(f"📊 天池O2O数据集 - 总用户数: {total_users}")
        for segment, count in segment_count.items():
            logger.info(f"👥 {segment}: {count} 人")
        logger.info("=" * 50)
        return segment_count

    def save_processed_data(self, df):
        os.makedirs(settings.DATA_PROCESSED_DIR, exist_ok=True)
        df.to_csv(self.processed_data_path, index=False, encoding="utf-8")

    def get_user_features(self, user_id):
        if not os.path.exists(self.processed_data_path):
            raw = self.load_raw_data()
            feat = self.preprocess(raw)
            self.save_processed_data(feat)
        else:
            feat = pd.read_csv(self.processed_data_path)

        user = feat[feat["User_id"] == int(user_id)]
        if user.empty:
            return {
                "historical_spend": 0,
                "avg_distance": 5.0,
                "is_new_user": True,
                "user_segment": "平台新用户"  # 默认分群
            }
        # 直接返回预计算的user_segment
        return user.iloc[0].to_dict()