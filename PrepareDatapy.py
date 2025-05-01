import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD
from typing import List, Dict, Tuple, Optional, Union

class DataProcessor:
    def __init__(self, name: str = "dataset"):
        self.name = name
        self.data_dict = {}
        self.encoders = {}
        self.embedding_models = {}
        self.feature_names = None
        
    def process_dataset(self, df: pd.DataFrame, 
                        categorical_cols: List[str] = None,
                        binary_cols: List[str] = None,
                        numerical_cols: List[str] = None,
                        target_col: str = None,
                        id_cols: List[str] = None,
                        scale_numerical: bool = True,
                        handle_missing: str = 'drop',
                        embedding_dim: int = 1) -> pd.DataFrame:
        
        if handle_missing == 'drop':
            df = df.dropna()
        elif handle_missing == 'impute':
            df = self._impute_missing_values(df)
            
        self.document_dataset(df, categorical_cols, binary_cols, numerical_cols, target_col, id_cols)
        
        processed_df = df.copy()
        
        if id_cols:
            processed_df = processed_df.drop(columns=id_cols)
        
        if target_col:
            y = processed_df[target_col].copy()
            X = processed_df.drop(columns=[target_col])
        else:
            y = None
            X = processed_df
            
        if binary_cols:
            for col in binary_cols:
                X = self._encode_binary_column(X, col)
        
        if categorical_cols:
            for col in categorical_cols:
                X = self._simple_embedding_encode_column(X, col, embedding_dim)
        
        if scale_numerical and numerical_cols:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
            self.encoders['numerical_scaler'] = scaler
        
        if target_col and target_col in df.columns:
            X[target_col] = y.values
        
        return X
    
    def _simple_embedding_encode_column(self, df: pd.DataFrame, column: str, 
                                       embedding_dim: int = 1) -> pd.DataFrame:
        print(f"Creating embedding for column: {column}")
        
        le = LabelEncoder()
        df[f"{column}_label"] = le.fit_transform(df[column].astype(str))
        self.encoders[column] = le
        
        one_hot = pd.get_dummies(df[column], prefix=column)
        
        if embedding_dim == 1 and one_hot.shape[1] <= 3:
            df[column] = df[f"{column}_label"]
            df = df.drop(columns=[f"{column}_label"])
            return df
        
        svd = TruncatedSVD(n_components=embedding_dim, random_state=42)
        embedding = svd.fit_transform(one_hot)
        self.embedding_models[column] = svd
        
        if embedding_dim == 1:
            df[column] = embedding.flatten()
        else:
            df = df.drop(columns=[column, f"{column}_label"])
            for i in range(embedding_dim):
                df[f"{column}_emb_{i}"] = embedding[:, i]
        
        return df
    
    def transform_new_data(self, df: pd.DataFrame, categorical_cols: List[str] = None,
                          binary_cols: List[str] = None, numerical_cols: List[str] = None,
                          scale_numerical: bool = True, embedding_dim: int = 1) -> pd.DataFrame:
        X = df.copy()
        
        if binary_cols:
            for col in binary_cols:
                if col in self.encoders:
                    X[col] = X[col].map(self.encoders[col])
        
        if categorical_cols:
            for col in categorical_cols:
                if col in self.embedding_models and col in self.encoders:
                    le = self.encoders[col]
                    
                    X[col] = X[col].astype(str)
                    X[col] = X[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                    
                    X[f"{col}_label"] = le.transform(X[col])
                    
                    one_hot = pd.get_dummies(X[col], prefix=col)
                    
                    expected_cols = [f"{col}_{c}" for c in le.classes_]
                    for expected_col in expected_cols:
                        if expected_col not in one_hot.columns:
                            one_hot[expected_col] = 0
                    
                    one_hot = one_hot[expected_cols]
                    
                    svd = self.embedding_models[col]
                    embedding = svd.transform(one_hot)
                    
                    if embedding_dim == 1:
                        X[col] = embedding.flatten()
                    else:
                        X = X.drop(columns=[col, f"{col}_label"])
                        for i in range(embedding_dim):
                            X[f"{col}_emb_{i}"] = embedding[:, i]
        
        if scale_numerical and numerical_cols and 'numerical_scaler' in self.encoders:
            scaler = self.encoders['numerical_scaler']
            X[numerical_cols] = scaler.transform(X[numerical_cols])
        
        return X
    
    def _encode_binary_column(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        unique_vals = df[column].unique()
        
        if len(unique_vals) > 2:
            print(f"Warning: Column {column} has more than 2 unique values: {unique_vals}")
            
        mapping = {val: i for i, val in enumerate(unique_vals)}
        self.encoders[column] = mapping
        
        df[column] = df[column].map(mapping)
        return df
    
    def _impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if df[col].dtype.kind in 'ifc':
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
        return df
        
    def document_dataset(self, df: pd.DataFrame, 
                         categorical_cols: List[str] = None,
                         binary_cols: List[str] = None,
                         numerical_cols: List[str] = None,
                         target_col: str = None,
                         id_cols: List[str] = None) -> Dict:
        
        if categorical_cols is None and binary_cols is None and numerical_cols is None:
            categorical_cols, binary_cols, numerical_cols = self._auto_detect_types(df)
        
        if categorical_cols:
            for col in categorical_cols:
                self._document_categorical(df, col)
        
        if binary_cols:
            for col in binary_cols:
                self._document_binary(df, col)
        
        if numerical_cols:
            for col in numerical_cols:
                self._document_numerical(df, col)
        
        if target_col:
            if df[target_col].nunique() <= 10:
                self._document_categorical(df, target_col, is_target=True)
            else:
                self._document_numerical(df, target_col, is_target=True)
        
        if id_cols:
            for col in id_cols:
                self.data_dict[col] = {'type': 'identifier', 'description': f'Unique identifier: {col}'}
        
        return self.data_dict
    
    def _document_categorical(self, df: pd.DataFrame, column: str, 
                             description: str = "", is_target: bool = False) -> Dict:
        unique_values = sorted(df[column].dropna().unique().tolist())
        value_counts = df[column].value_counts().to_dict()
        
        mapping = {i: val for i, val in enumerate(unique_values)}
        reverse_mapping = {val: i for i, val in enumerate(unique_values)}
        
        self.data_dict[column] = {
            'type': 'categorical',
            'description': description or f"Categorical variable: {column}",
            'is_target': is_target,
            'unique_values': unique_values,
            'value_counts': value_counts,
            'mapping': mapping,
            'reverse_mapping': reverse_mapping,
            'n_unique': len(unique_values),
            'missing_count': int(df[column].isna().sum())
        }
        
        self.encoders[column] = reverse_mapping
        
        return self.data_dict[column]
    
    def _document_binary(self, df: pd.DataFrame, column: str, 
                        description: str = "", is_target: bool = False) -> Dict:
        unique_values = sorted(df[column].dropna().unique().tolist())
        value_counts = df[column].value_counts().to_dict()
        
        if len(unique_values) != 2:
            print(f"Warning: Binary column {column} has {len(unique_values)} unique values.")
        
        mapping = {unique_values[0]: 0, unique_values[1]: 1} if len(unique_values) >= 2 else {}
        
        self.data_dict[column] = {
            'type': 'binary',
            'description': description or f"Binary variable: {column}",
            'is_target': is_target,
            'unique_values': unique_values,
            'value_counts': value_counts,
            'mapping': mapping,
            'n_unique': len(unique_values),
            'missing_count': int(df[column].isna().sum())
        }
        
        self.encoders[column] = mapping
        
        return self.data_dict[column]
    
    def _document_numerical(self, df: pd.DataFrame, column: str, 
                           description: str = "", is_target: bool = False) -> Dict:
        stats = df[column].describe().to_dict()
        
        self.data_dict[column] = {
            'type': 'numerical',
            'description': description or f"Numerical variable: {column}",
            'is_target': is_target,
            'stats': stats,
            'missing_count': int(df[column].isna().sum())
        }
        
        return self.data_dict[column]
    
    def _auto_detect_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
        categorical_cols = []
        binary_cols = []
        numerical_cols = []
        
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                if df[col].nunique() <= 2:
                    binary_cols.append(col)
                else:
                    categorical_cols.append(col)
            elif df[col].dtype.kind in 'ifc':
                if df[col].nunique() <= 2:
                    binary_cols.append(col)
                else:
                    numerical_cols.append(col)
        
        return categorical_cols, binary_cols, numerical_cols
    
    def get_dataset_info(self) -> Dict:
        return self.data_dict
