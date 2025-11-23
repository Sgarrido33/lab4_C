import pandas as pd
import numpy as np
from datetime import datetime
import logging
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('feature-engineering')

def create_features(df):
    """Create new features from existing data."""
    logger.info("Creating new features")
    df_featured = df.copy()

    current_year = datetime.now().year
    df_featured['house_age'] = current_year - df_featured['year_built']
    
    df_featured['price_per_sqft'] = df_featured['price'] / df_featured['sqft']
    
    df_featured['bed_bath_ratio'] = df_featured['bedrooms'] / df_featured['bathrooms']
    df_featured['bed_bath_ratio'] = df_featured['bed_bath_ratio'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return df_featured

def create_preprocessor():
    """Create a preprocessing pipeline."""
    categorical_features = ['location', 'condition']
    numerical_features = ['sqft', 'bedrooms', 'bathrooms', 'house_age', 'price_per_sqft', 'bed_bath_ratio']

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean'))
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        verbose_feature_names_out=False # ESTO AYUDA A LIMPIAR NOMBRES EN VERSIONES MODERNAS
    )

    return preprocessor

def run_feature_engineering(input_file, output_file, preprocessor_file):
    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)

    df_featured = create_features(df)
    
    preprocessor = create_preprocessor()
    X = df_featured.drop(columns=['price'], errors='ignore')
    y = df_featured['price'] if 'price' in df_featured.columns else None
    
    # Fit and transform
    X_transformed = preprocessor.fit_transform(X)
    logger.info("Fitted the preprocessor")

    joblib.dump(preprocessor, preprocessor_file)

    # --- LÓGICA DE LIMPIEZA DE NOMBRES ---
    # Obtenemos los nombres brutos
    try:
        raw_features = preprocessor.get_feature_names_out()
    except:
        # Fallback si falla sklearn
        raw_features = [f"feat_{i}" for i in range(X_transformed.shape[1])]

    # Limpiamos prefijos molestos (ej: 'num__sqft' -> 'sqft', 'cat__location_Suburb' -> 'location_Suburb')
    clean_features = []
    for feat in raw_features:
        if "__" in feat:
            clean_features.append(feat.split("__")[-1])
        else:
            clean_features.append(feat)

    logger.info(f"Feature names generated: {clean_features}")

    df_transformed = pd.DataFrame(X_transformed, columns=clean_features)
    
    if y is not None:
        df_transformed['price'] = y.values
        
    df_transformed.to_csv(output_file, index=False)
    logger.info(f"Saved fully preprocessed data to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--preprocessor', required=True)
    args = parser.parse_args()

    run_feature_engineering(args.input, args.output, args.preprocessor)