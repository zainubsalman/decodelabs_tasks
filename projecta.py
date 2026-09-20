"""
DecodeLabs Project 1: Advanced EDA & Feature Engineering
==========================================================

This script follows the three phases from the project brief:

PHASE 1 - Securing Input Fidelity
    - Handle missing values using the Missing Data Decision Matrix
    - Detect and cap outliers using the IQR method (Winsorization)

PHASE 2 - The Vectorized Computation Engine
    - One-Hot Encode categorical columns (avoids Label Encoding distortion)
    - Detect and remove multicollinear features (correlation > 0.80)

PHASE 3 - Structural Contracts and Scaling
    - Validate the final dataset against a Pandera schema (data contract)

Replace the sample dataset at the bottom with your real dataset to use this
on your actual project data.
"""

import numpy as np
import pandas as pd
import pandera as pa
from pandera import Column, Check, DataFrameSchema
from sklearn.impute import KNNImputer


# ============================================================
# PHASE 1: SECURING INPUT FIDELITY
# ============================================================

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the Missing Data Decision Matrix column by column:

    Missing %       Action
    ---------       ------
    < 5%            Drop the rows with missing values in that column
    5% - 20%        Statistical imputation:
                        - numeric column  -> fill with the column median
                        - text/category   -> fill with the most common value
    > 20%           KNN Imputation (numeric columns only), which looks at
                     similar rows to estimate a realistic value.
    """
    df = df.copy()
    missing_pct = df.isnull().mean() * 100

    for col in df.columns:
        pct = missing_pct[col]

        if pct == 0:
            continue  # nothing missing, skip

        elif pct < 5:
            df = df.dropna(subset=[col])

        elif pct <= 20:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])

        else:  # > 20% missing
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                imputer = KNNImputer(n_neighbors=5)
                df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            else:
                # KNN imputation only works cleanly on numeric data,
                # so for a heavily-missing text column, fall back to mode.
                df[col] = df[col].fillna(df[col].mode()[0])

    return df


def cap_outliers_iqr(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    """
    Detect outliers with the IQR method and cap (Winsorize) them instead
    of deleting rows, so no data volume is lost.

        Lower Bound = Q1 - 1.5 * IQR
        Upper Bound = Q3 + 1.5 * IQR
    """
    df = df.copy()
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns

    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.clip(df[col], lower_bound, upper_bound)

    return df


# ============================================================
# PHASE 2: THE VECTORIZED COMPUTATION ENGINE
# ============================================================

def one_hot_encode(df: pd.DataFrame, categorical_cols=None) -> pd.DataFrame:
    """
    Convert text/category columns into One-Hot Encoded columns.

    This avoids the "Label Encoding flaw": assigning 1, 2, 3 to categories
    implies a false mathematical distance (e.g. Tokyo = 3x London), which
    One-Hot Encoding fixes by giving each category its own 0/1 column.
    """
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    return pd.get_dummies(df, columns=list(categorical_cols), drop_first=True)


def remove_multicollinearity(df: pd.DataFrame, target_col: str, threshold: float = 0.80) -> pd.DataFrame:
    """
    Find feature pairs with correlation above the threshold (default 0.80).
    For each such pair, keep whichever feature is more correlated with the
    target variable, and drop the weaker one.
    """
    df = df.copy()
    feature_cols = [c for c in df.columns if c != target_col]

    # Only correlate numeric columns
    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_features].corr().abs()

    # Step 1 & 2: build the matrix, isolate the upper triangle
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = set()
    # Step 3 & 4: find pairs > threshold, compare each to the target, drop the weaker
    for col in upper_triangle.columns:
        for row in upper_triangle.index:
            value = upper_triangle.loc[row, col]
            if pd.notna(value) and value > threshold:
                corr_row_target = abs(df[row].corr(df[target_col]))
                corr_col_target = abs(df[col].corr(df[target_col]))
                weaker = row if corr_row_target < corr_col_target else col
                to_drop.add(weaker)

    if to_drop:
        print(f"  Dropping collinear features: {sorted(to_drop)}")

    return df.drop(columns=list(to_drop))


# ============================================================
# PHASE 3: STRUCTURAL CONTRACTS AND SCALING
# ============================================================

def build_schema(df: pd.DataFrame, target_col: str) -> DataFrameSchema:
    """
    Build a simple Pandera schema that checks:
      - every numeric column has no nulls left
      - every numeric column is a real float/int (not text hiding as a number)

    This acts as a "data contract": a rulebook the dataset must pass
    before being handed to a model.
    """
    columns = {}
    for col in df.columns:
        if pd.api.types.is_bool_dtype(df[col]):
            columns[col] = Column(bool, nullable=False)
        elif pd.api.types.is_numeric_dtype(df[col]):
            columns[col] = Column(float, Check(lambda s: s.notna().all()), nullable=False)
        else:
            columns[col] = Column(object, nullable=False)

    return DataFrameSchema(columns)


def validate_schema(df: pd.DataFrame, schema: DataFrameSchema):
    """
    Validate the dataframe against the schema.
    lazy=True means Pandera checks the WHOLE dataframe and reports every
    failure at once, instead of stopping at the first error.
    """
    try:
        validated_df = schema.validate(df, lazy=True)
        print("Schema validation passed. Data is contract-safe.")
        return validated_df
    except pa.errors.SchemaErrors as err:
        print("Schema validation FAILED. Failure cases:")
        print(err.failure_cases)
        return None


# ============================================================
# FULL PIPELINE (runs all phases in order)
# ============================================================

def run_pipeline(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    print("Step 1: Handling missing values...")
    df = handle_missing_values(df)

    print("Step 2: Capping outliers with IQR (Winsorization)...")
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]
    df = cap_outliers_iqr(df, columns=numeric_cols)

    print("Step 3: One-Hot Encoding categorical columns...")
    df = one_hot_encode(df)

    print("Step 4: Removing multicollinear features (correlation > 0.80)...")
    df = remove_multicollinearity(df, target_col=target_col)

    print("Step 5: Validating final dataset against schema contract...")
    schema = build_schema(df, target_col)
    df = validate_schema(df, schema)

    print(f"\nPipeline complete. Final shape: {df.shape}")
    return df


# ============================================================
# DEMO RUN (replace this section with your real dataset)
# ============================================================

if __name__ == "__main__":
    # --- Replace these lines with your real data ---
    # df = pd.read_csv("your_dataset.csv")
    # cleaned_df = run_pipeline(df, target_col="your_target_column_name")

    # Sample data below just so this script runs end-to-end as a demo
    np.random.seed(42)
    n = 200

    sample_df = pd.DataFrame({
        "age": np.random.normal(35, 10, n),
        "income": np.random.normal(50000, 15000, n),
        "city": np.random.choice(["Lahore", "Karachi", "Islamabad"], n),
        "target": np.random.normal(100, 20, n),
    })

    # simulate a collinear column (income_copy is basically = income)
    sample_df["income_copy"] = sample_df["income"] * 1.02 + np.random.normal(0, 500, n)

    # simulate some missing values
    sample_df.loc[0:5, "age"] = np.nan          # < 5% missing -> will be dropped
    sample_df.loc[10:25, "income"] = np.nan     # ~8% missing -> median imputation

    # simulate an outlier
    sample_df.loc[30, "income"] = 900000

    print("Original shape:", sample_df.shape)
    print(sample_df.head(), "\n")

    cleaned_df = run_pipeline(sample_df, target_col="target")
    print("\nFinal cleaned data preview:")
    print(cleaned_df.head())