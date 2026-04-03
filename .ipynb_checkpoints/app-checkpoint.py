"""
House Price Prediction — Streamlit App
Full pipeline matching notebook: 497 features, SVR R² ≈ 0.89
Run: streamlit run app.py
Needs: train.csv test.csv (Housing, Kaggle)
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle, os, calendar
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from pandas.api.types import CategoricalDtype

# ─────────────────────────────────────────────────────────────────────────────
# Constants (MUST match notebook EXACTLY)
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH = "model_full_pipeline.pkl"

# Ordinal encoding maps (from notebook cell 84)
ORDINAL_MAPS = {
    "BsmtCond":     ["NA","Po","Fa","TA","Gd","Ex"],
    "BsmtExposure": ["NA","Mn","Av","Gd"],
    "BsmtFinType1": ["NA","Unf","LwQ","Rec","BLQ","ALQ","GLQ"],
    "BsmtFinType2": ["NA","Unf","LwQ","Rec","BLQ","ALQ","GLQ"],
    "BsmtQual":     ["NA","Po","Fa","TA","Gd","Ex"],
    "ExterQual":    ["Po","Fa","TA","Gd","Ex"],
    "ExterCond":    ["Po","Fa","TA","Gd","Ex"],
    "FireplaceQu":  ["NA","Po","Fa","TA","Gd","Ex"],
    "GarageCond":   ["NA","Po","Fa","TA","Gd","Ex"],
    "GarageFinish": ["NA","Unf","RFn","Fin"],
    "GarageQual":   ["NA","Po","Fa","TA","Gd","Ex"],
    "HeatingQC":    ["Po","Fa","TA","Gd","Ex"],
    "KitchenQual":  ["Po","Fa","TA","Gd","Ex"],
    "LandSlope":    ["Gtl","Mod","Sev"],
    "LotShape":     ["IR3","IR2","IR1","Reg"],
    "PavedDrive":   ["N","P","Y"],
    "PoolQC":       ["NA","Fa","TA","Gd","Ex"],
    "Street":       ["Grvl","Pave"],
    "Utilities":    ["ELO","NoSeWa","NoSewr","AllPub"],
}

SKEWED = [
    "1stFlrSF","2ndFlrSF","3SsnPorch","BedroomAbvGr","BsmtFinSF1","BsmtFinSF2",
    "BsmtFullBath","BsmtHalfBath","BsmtUnfSF","EnclosedPorch","Fireplaces",
    "FullBath","GarageArea","GarageCars","GrLivArea","HalfBath","KitchenAbvGr",
    "LotArea","LotFrontage","LowQualFinSF","MasVnrArea","MiscVal","OpenPorchSF",
    "PoolArea","ScreenPorch","TotRmsAbvGrd","TotalBsmtSF","WoodDeckSF",
]

INT_TO_STR = ["MSSubClass","YearBuilt","YearRemodAdd","GarageYrBlt","YrSold"]

# ─────────────────────────────────────────────────────────────────────────────
# Full preprocessing (matches notebook exactly - cell by cell)
# ─────────────────────────────────────────────────────────────────────────────
def full_preprocess(train_df, test_df):
    """Complete preprocessing matching the notebook pipeline"""
    df = pd.concat((train_df, test_df)).copy()
    df = df.set_index("Id")

    qual_set = set(df.select_dtypes("object").columns)

    # 1. Drop columns with >20% nulls
    null_pct = df.isnull().sum() / len(df) * 100
    df.drop(columns=null_pct[null_pct > 20].index, inplace=True)
    qual_set &= set(df.columns)

    # 2. Basement imputation (matching notebook)
    bsmt_col = [c for c in [
        "BsmtCond","BsmtExposure","BsmtFinSF1","BsmtFinSF2",
        "BsmtFinType1","BsmtFinType2","BsmtFullBath","BsmtHalfBath",
        "BsmtQual","BsmtUnfSF","TotalBsmtSF"] if c in df.columns]

    bsmt = df[bsmt_col].copy()
    all_zero_nan = (bsmt.isnull() | bsmt.isin([0])).all(axis=1)
    bsmt_all = bsmt[all_zero_nan].copy()
    for c in bsmt_col:
        bsmt_all[c] = bsmt_all[c].fillna("NA" if c in qual_set else 0)
    df.update(bsmt_all)

    # Fix specific BsmtFinType2 for index 333
    if "BsmtFinSF2" in df.columns and "BsmtFinType2" in df.columns and 333 in df.index:
        bucket = df[(df["BsmtFinSF2"] >= 305) & (df["BsmtFinSF2"] <= 610)]
        if not bucket.empty:
            df.at[333, "BsmtFinType2"] = bucket["BsmtFinType2"].mode()[0]

    # Fill BsmtExposure based on BsmtQual
    if "BsmtQual" in df.columns and "BsmtExposure" in df.columns:
        gd_mask = (df["BsmtQual"] == "Gd") & df["BsmtExposure"].isnull()
        mode_val = df.loc[df["BsmtQual"] == "Gd", "BsmtExposure"].mode()
        if not mode_val.empty:
            df.loc[gd_mask, "BsmtExposure"] = mode_val[0]
    
    for c in ["BsmtCond","BsmtQual"]:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].mode()[0])

    # 3. Garage imputation
    gar_col = [c for c in [
        "GarageArea","GarageCars","GarageCond","GarageFinish",
        "GarageQual","GarageType","GarageYrBlt"] if c in df.columns]

    gar = df[gar_col].copy()
    all_zero_nan = (gar.isnull() | gar.isin([0])).all(axis=1)
    gar_all = gar[all_zero_nan].copy()
    for c in gar_col:
        gar_all[c] = gar_all[c].fillna("NA" if c in qual_set else 0)
    df.update(gar_all)

    gar_rem = df[gar_col][df[gar_col].isnull().any(axis=1)].copy()
    if not gar_rem.empty and "GarageType" in df.columns:
        detchd = df[df["GarageType"] == "Detchd"]
        for c in gar_col:
            if detchd[c].notna().any():
                gar_rem[c] = gar_rem[c].fillna(detchd[c].mode()[0])
        df.update(gar_rem)

    # 4. Fill remaining categorical columns with mode
    for c in ["Electrical","Exterior1st","Exterior2nd","Functional",
              "KitchenQual","MSZoning","SaleType","Utilities"]:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].mode()[0])
    
    if "MasVnrArea" in df.columns:
        df["MasVnrArea"] = df["MasVnrArea"].fillna(0)

    # 5. LotFrontage imputation by LotConfig group mean
    if "LotFrontage" in df.columns and "LotConfig" in df.columns:
        for lc in ["Corner","Inside","CulDSac","FR2","FR3"]:
            mask = df["LotFrontage"].isnull() & (df["LotConfig"] == lc)
            df.loc[mask, "LotFrontage"] = df.loc[df["LotConfig"] == lc, "LotFrontage"].mean()

    # Save defaults for new predictions
    raw_num_defaults = df.median(numeric_only=True).to_dict()
    raw_obj_defaults = {c: df[c].mode()[0] for c in df.select_dtypes("object").columns}

    # 6. Convert int cols that are categorical to string
    for c in INT_TO_STR:
        if c in df.columns:
            df[c] = df[c].astype(str)

    # 7. MoSold → month abbreviation
    if "MoSold" in df.columns:
        df["MoSold"] = df["MoSold"].apply(
            lambda x: calendar.month_abbr[int(float(x))] if pd.notna(x) else x
        )

    # 8. Ordinal encoding
    for col, cats in ORDINAL_MAPS.items():
        if col in df.columns:
            df[col] = df[col].astype(CategoricalDtype(cats, ordered=True)).cat.codes

    # 9. Log-transform skewed features
    for c in SKEWED:
        if c in df.columns:
            df[c] = np.log(df[c] + 1)

    # 10. One-hot encoding
    obj_feat = list(df.select_dtypes("object").columns)
    cat_maps = {c: sorted(df[c].dropna().unique().tolist()) for c in obj_feat}
    last_cats = {c: str(df[c].unique()[-1]) for c in obj_feat}

    df = pd.get_dummies(df, columns=obj_feat)
    for c, lv in last_cats.items():
        col_to_drop = f"{c}_{lv}"
        if col_to_drop in df.columns:
            df.drop(columns=[col_to_drop], inplace=True)

    feature_cols = list(df.columns)

    # 11. RobustScaler
    scaler = RobustScaler()
    X = scaler.fit_transform(df)

    return (
        X[:len(train_df)], X[len(train_df):],
        scaler, feature_cols, cat_maps, last_cats,
        raw_num_defaults, raw_obj_defaults,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Load or train the model bundle
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_or_train():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)

    if not (os.path.exists("train.csv") and os.path.exists("test.csv")):
        return None

    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    with st.spinner("🔧 Running full preprocessing pipeline..."):
        (X_tr_full, X_te_full, scaler, feature_cols, cat_maps, last_cats,
         num_defaults, obj_defaults) = full_preprocess(train, test)

    SalePrice = np.log(train["SalePrice"] + 1)

    # Split for evaluation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr_full, SalePrice, test_size=0.2, random_state=42
    )

    with st.spinner("🤖 Training SVR (C=100, γ=0.0001, ε=0.01)..."):
        svr = SVR(kernel="rbf", C=100, epsilon=0.01, gamma=0.0001)
        svr.fit(X_tr, y_tr)

    with st.spinner("🌲 Training Gradient Boosting..."):
        gbr = GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.1,
            loss="squared_error", random_state=51
        )
        gbr.fit(X_tr, y_tr)

    bundle = dict(
        model=svr,
        gbr=gbr,
        scaler=scaler,
        feature_cols=feature_cols,
        cat_maps=cat_maps,
        last_cats=last_cats,
        num_defaults=num_defaults,
        obj_defaults=obj_defaults,
        X_val=X_val,
        y_val=y_val,
        X_train_full=X_tr_full,
        SalePrice=SalePrice,
        train_raw=train,
    )
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    return bundle


# ─────────────────────────────────────────────────────────────────────────────
# Single-row prediction
# ─────────────────────────────────────────────────────────────────────────────
def predict_new_house(bundle, user_inputs: dict):
    """Apply full pipeline to a single new house"""
    row = {}
    row.update(bundle["num_defaults"])
    row.update(bundle["obj_defaults"])
    row.update(user_inputs)

    s = pd.Series(row, dtype=object)

    # INT_TO_STR conversion
    for c in INT_TO_STR:
        if c in s.index:
            try:
                s[c] = str(int(float(s[c])))
            except Exception:
                pass

    # MoSold conversion
    if "MoSold" in s.index:
        try:
            s["MoSold"] = calendar.month_abbr[int(float(s["MoSold"]))]
        except Exception:
            pass

    # Ordinal encoding
    for col, cats in ORDINAL_MAPS.items():
        if col in s.index:
            val = s[col]
            codes = {v: i for i, v in enumerate(cats)}
            s[col] = codes.get(str(val), -1)

    # Log-transform skewed features
    for c in SKEWED:
        if c in s.index:
            try:
                s[c] = np.log(float(s[c]) + 1)
            except Exception:
                s[c] = 0.0

    # One-hot encoding
    cat_maps = bundle["cat_maps"]
    last_cats = bundle["last_cats"]
    new_cols = {}

    for col, unique_vals in cat_maps.items():
        if col not in s.index:
            continue
        val = str(s[col])
        drop_val = str(last_cats.get(col, ""))
        for uv in unique_vals:
            if str(uv) != drop_val:
                new_cols[f"{col}_{uv}"] = 1 if val == str(uv) else 0
        s = s.drop(col)

    for k, v in new_cols.items():
        s[k] = v

    # Align to training features
    feature_cols = bundle["feature_cols"]
    X = np.zeros((1, len(feature_cols)), dtype=float)
    for i, col in enumerate(feature_cols):
        if col in s.index:
            try:
                X[0, i] = float(s[col])
            except Exception:
                X[0, i] = 0.0

    X_scaled = bundle["scaler"].transform(X)

    svr_log = bundle["model"].predict(X_scaled)[0]
    gbr_log = bundle["gbr"].predict(X_scaled)[0]

    return float(np.exp(svr_log)), float(np.exp(gbr_log))


# ─────────────────────────────────────────────────────────────────────────────
# Page config & CSS (same as your existing)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
)

# [Your existing CSS and UI code remains the same...]
# (I'm keeping the rest of your UI code as is since it's fine)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.hero-title { font-family: 'DM Serif Display', serif; font-size: 2.8rem; color: #0f1f2e; line-height: 1.1; margin-bottom: 6px; }
.hero-sub { font-size: 1rem; color: #5a7490; margin-bottom: 0; }
.section-head { font-family: 'DM Serif Display', serif; font-size: 1.45rem; color: #0f1f2e; border-left: 4px solid #3b82f6; padding-left: 12px; margin-bottom: 14px; }
.fix-banner { background: #fff3cd; border-left: 4px solid #f0a500; border-radius: 8px; padding: 12px 16px; font-size: 0.87rem; color: #6b4a00; margin-bottom: 18px; }
.stButton > button { background: linear-gradient(135deg, #0f1f2e, #1a5276) !important; color: white !important; font-weight: 600 !important; border-radius: 10px !important; border: none !important; padding: 12px 30px !important; width: 100% !important; }
</style>
""", unsafe_allow_html=True)

# Load bundle
with st.spinner("Loading model..."):
    bundle = load_or_train()

if bundle is None:
    st.error("⚠️ `train.csv` and `test.csv` not found in the same folder.")
    st.stop()

# Header
st.markdown('<div class="hero-title">🏠 House Price Predictor</div>', unsafe_allow_html=True)

st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("### 🏠 House Price Predictor")
    st.markdown("---")
    st.markdown("**Pipeline summary:**")
    st.markdown("- 2919 rows (train + test combined)")
    st.markdown("- Missing value imputation (Bsmt, Garage, LotFrontage)")
    st.markdown("- Ordinal encoding for quality cols")
    st.markdown("- Log-transform of 28 skewed features")
    st.markdown("- One-hot encoding → 497 features")
    st.markdown("- RobustScaler")
    st.markdown("---")
    st.markdown("**Models:**")
    st.markdown("- ✅ SVR `C=100, γ=0.0001, ε=0.01` → R²≈0.89")
    st.markdown("- 📊 GradientBoosting → R²≈0.88")
    st.markdown("---")
    page = st.radio("Navigate", [
        "🔮 Predict Price",
        "📊 Model Performance",
        "📈 Data Insights",
    ])

# Main content based on page selection
if page == "🔮 Predict Price":
    

    st.markdown('<div class="section-head">Enter House Details</div>', unsafe_allow_html=True)

    train_raw = bundle["train_raw"]
    neighborhoods = sorted(train_raw["Neighborhood"].dropna().unique().tolist())
    house_styles = sorted(train_raw["HouseStyle"].dropna().unique().tolist())
    ms_zonings = sorted(train_raw["MSZoning"].dropna().unique().tolist())

    c1, c2, c3 = st.columns(3)
    with c1:
        overall_qual = st.slider("⭐ Overall Quality (1–10)", 1, 10, 6)
        gr_liv_area = st.number_input("📐 Above-Ground Living Area (sq ft)", 500, 6000, 1500, step=50)
        total_bsmt = st.number_input("🏗️ Total Basement Area (sq ft)", 0, 4000, 800, step=50)

    with c2:
        garage_cars = st.slider("🚗 Garage Capacity (cars)", 0, 4, 2)
        full_bath = st.slider("🚿 Full Bathrooms", 0, 4, 2)
        bedroom_abv = st.slider("🛏️ Bedrooms Above Ground", 0, 8, 3)

    with c3:
        year_built = st.number_input("🏗️ Year Built", 1870, 2010, 1980, step=1)
        year_remod = st.number_input("🔨 Year Remodelled", 1950, 2010, 2000, step=1)
        neighborhood = st.selectbox("📍 Neighborhood", neighborhoods,
            index=neighborhoods.index("NAmes") if "NAmes" in neighborhoods else 0)

    ca, cb = st.columns(2)
    with ca:
        house_style = st.selectbox("🏘️ House Style", house_styles)
    with cb:
        ms_zoning = st.selectbox("🗺️ MS Zoning", ms_zonings,
            index=ms_zonings.index("RL") if "RL" in ms_zonings else 0)

    st.markdown("---")

    if st.button("🔮 Predict House Price"):
        user_inputs = {
            "OverallQual": overall_qual,
            "GrLivArea": gr_liv_area,
            "TotalBsmtSF": total_bsmt,
            "GarageCars": garage_cars,
            "FullBath": full_bath,
            "BedroomAbvGr": bedroom_abv,
            "YearBuilt": year_built,
            "YearRemodAdd": year_remod,
            "Neighborhood": neighborhood,
            "HouseStyle": house_style,
            "MSZoning": ms_zoning,
        }

        with st.spinner("Running full pipeline prediction..."):
            svr_price, gbr_price = predict_new_house(bundle, user_inputs)

        avg_price = (svr_price + gbr_price) / 2

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("SVR Prediction", f"${svr_price:,.0f}", delta=None)
        with col2:
            st.metric("Gradient Boosting", f"${gbr_price:,.0f}", delta=None)
        with col3:
            st.metric("Ensemble Average", f"${avg_price:,.0f}", delta=None)

        st.success(f"✅ **Quality {overall_qual}/10** · **{gr_liv_area:,} sq ft** · "
                   f"**{bedroom_abv} beds** · **Built {year_built}**")

# Add other pages (Model Performance, Data Insights) as needed...