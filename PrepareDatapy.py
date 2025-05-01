#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns


class DataProcessor:
    def __init__(self, name: str = "dataset"):
        self.name = name
        self.data_dict = {}
        self.encoders = {}
        self.column_transformer = None
        self.feature_names = None
        
    def process_dataset(self, df: pd.DataFrame, 
                        categorical_cols: List[str] = None,
                        binary_cols: List[str] = None,
                        numerical_cols: List[str] = None,
                        target_col: str = None,
                        id_cols: List[str] = None,
                        scale_numerical: bool = True,
                        handle_missing: str = 'drop') -> pd.DataFrame:
        
        # Handle missing values
        if handle_missing == 'drop':
            df = df.dropna()
        elif handle_missing == 'impute':
            df = self._impute_missing_values(df)
            
        # Document the dataset
        self.document_dataset(df, categorical_cols, binary_cols, numerical_cols, target_col, id_cols)
        
        # Create a copy to avoid modifying the original
        processed_df = df.copy()
        
        # Handle ID columns
        if id_cols:
            processed_df = processed_df.drop(columns=id_cols)
        
        # Separate target if provided
        if target_col:
            y = processed_df[target_col].copy()
            X = processed_df.drop(columns=[target_col])
        else:
            y = None
            X = processed_df
            
        # Process binary variables
        if binary_cols:
            for col in binary_cols:
                X = self._encode_binary_column(X, col)
            
        # Build and apply transformers
        self._build_transformer(scale_numerical)
        X_transformed = self._apply_transformer(X)
        
        # Combine with target if needed
        if target_col:
            X_transformed[target_col] = y
            
        return X_transformed
    
    def _impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if df[col].dtype.kind in 'ifc':  # integer, float, complex
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
        
        # Auto-detect column types if not specified
        if categorical_cols is None and binary_cols is None and numerical_cols is None:
            categorical_cols, binary_cols, numerical_cols = self._auto_detect_types(df)
        
        # Document each type
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
            raise ValueError(f"Column {column} has {len(unique_values)} unique values, expected 2 for binary")
        
        false_label = unique_values[0]
        true_label = unique_values[1]
        
        mapping = {0: false_label, 1: true_label}
        reverse_mapping = {false_label: 0, true_label: 1}
        
        self.data_dict[column] = {
            'type': 'binary',
            'description': description or f"Binary variable: {column}",
            'is_target': is_target,
            'unique_values': unique_values,
            'value_counts': value_counts,
            'mapping': mapping,
            'reverse_mapping': reverse_mapping,
            'missing_count': int(df[column].isna().sum())
        }
        
        self.encoders[column] = reverse_mapping
        
        return self.data_dict[column]
    
    def _document_numerical(self, df: pd.DataFrame, column: str, 
                           description: str = "", is_target: bool = False) -> Dict:
        stats = df[column].describe().to_dict()
        
        self.data_dict[column] = {
            'type': 'numerical',
            'description': description or f"Numerical variable: {column}",
            'is_target': is_target,
            'min': float(stats['min']),
            'max': float(stats['max']),
            'mean': float(stats['mean']),
            'median': float(df[column].median()),
            'std': float(stats['std']),
            'missing_count': int(df[column].isna().sum()),
            'percentiles': {
                '25%': float(stats['25%']),
                '50%': float(stats['50%']),
                '75%': float(stats['75%'])
            }
        }
        
        return self.data_dict[column]
    
    def _auto_detect_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
        categorical_cols = []
        binary_cols = []
        numerical_cols = []
        
        for col in df.columns:
            if df[col].nunique() > df.shape[0] * 0.5:
                continue
                
            if df[col].nunique() <= 2:
                binary_cols.append(col)
            elif df[col].dtype == 'object' or df[col].nunique() < 20:
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)
                
        return categorical_cols, binary_cols, numerical_cols
    
    def _encode_binary_column(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        if column not in self.data_dict:
            raise ValueError(f"Column {column} not found in data dictionary")
            
        if self.data_dict[column]['type'] != 'binary':
            raise ValueError(f"Column {column} is not a binary variable")
            
        reverse_mapping = self.data_dict[column].get('reverse_mapping')
        
        if reverse_mapping:
            df[column] = df[column].map(reverse_mapping)
        else:
            unique_values = sorted(df[column].dropna().unique().tolist())
            if len(unique_values) != 2:
                raise ValueError(f"Binary column {column} should have exactly 2 unique values")
                
            df[column] = df[column].map({unique_values[0]: 0, unique_values[1]: 1})
                
        return df
    
    def _build_transformer(self, scale_numerical: bool = True):
        transformers = []
        
        categorical_cols = [col for col, info in self.data_dict.items() 
                           if info['type'] == 'categorical' and not info.get('is_target', False)]
        binary_cols = [col for col, info in self.data_dict.items() 
                      if info['type'] == 'binary' and not info.get('is_target', False)]
        numerical_cols = [col for col, info in self.data_dict.items() 
                         if info['type'] == 'numerical' and not info.get('is_target', False)]
        
        if categorical_cols:
            categorical_transformer = Pipeline(steps=[
                ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
            ])
            transformers.append(('cat', categorical_transformer, categorical_cols))
        
        if binary_cols:
            transformers.append(('bin', 'passthrough', binary_cols))
        
        if numerical_cols:
            if scale_numerical:
                numerical_transformer = Pipeline(steps=[
                    ('scaler', StandardScaler())
                ])
                transformers.append(('num', numerical_transformer, numerical_cols))
            else:
                transformers.append(('num', 'passthrough', numerical_cols))
        
        self.column_transformer = ColumnTransformer(
            transformers=transformers,
            remainder='drop'
        )
        
        self._set_feature_names(categorical_cols, binary_cols, numerical_cols)
        
        return self.column_transformer
    
    def _set_feature_names(self, categorical_cols, binary_cols, numerical_cols):
        feature_names = []
        
        for col in categorical_cols:
            unique_values = self.data_dict[col]['unique_values']
            for val in unique_values:
                feature_names.append(f"{col}_{val}")
        
        feature_names.extend(binary_cols)
        feature_names.extend(numerical_cols)
        
        self.feature_names = feature_names
    
    def _apply_transformer(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.column_transformer is None:
            raise ValueError("Transformer not built yet. Call _build_transformer first.")
        
        X_transformed = self.column_transformer.fit_transform(X)
        X_df = pd.DataFrame(X_transformed, index=X.index, columns=self.feature_names)
        
        return X_df
    
    def save_documentation(self, filepath: str = None):
        if filepath is None:
            filepath = f"{self.name}_data_dictionary.json"
            
        with open(filepath, 'w') as f:
            json.dump(self.data_dict, f, indent=2, default=str)
    
    def load_documentation(self, filepath: str):
        with open(filepath, 'r') as f:
            self.data_dict = json.load(f)
            
        self.encoders = {}
        for col, info in self.data_dict.items():
            if 'reverse_mapping' in info:
                self.encoders[col] = info['reverse_mapping']
    
    def generate_report(self) -> str:
        report = f"# {self.name.title()} Dataset Documentation\n\n"
        
        targets = [col for col, info in self.data_dict.items() if info.get('is_target', False)]
        if targets:
            report += "## Target Variable\n\n"
            for col in targets:
                info = self.data_dict[col]
                report += f"### {col}\n"
                report += f"- Type: {info['type']}\n"
                if info['type'] == 'numerical':
                    report += f"- Range: {info['min']} - {info['max']}\n"
                    report += f"- Mean: {info['mean']:.2f}\n"
                    report += f"- Median: {info['median']:.2f}\n"
                else:
                    report += f"- Unique values: {info['unique_values']}\n"
                    report += f"- Distribution: {info['value_counts']}\n"
                report += "\n"
        
        categorical_cols = [col for col, info in self.data_dict.items() 
                           if info['type'] == 'categorical' and not info.get('is_target', False)]
        if categorical_cols:
            report += "## Categorical Variables\n\n"
            for col in categorical_cols:
                info = self.data_dict[col]
                report += f"### {col}\n"
                report += f"- Unique values ({info['n_unique']}): {info['unique_values']}\n"
                report += f"- Missing values: {info['missing_count']}\n\n"
        
        binary_cols = [col for col, info in self.data_dict.items() 
                      if info['type'] == 'binary' and not info.get('is_target', False)]
        if binary_cols:
            report += "## Binary Variables\n\n"
            for col in binary_cols:
                info = self.data_dict[col]
                report += f"### {col}\n"
                report += f"- Values: {info['unique_values'][0]} (0), {info['unique_values'][1]} (1)\n"
                report += f"- Missing values: {info['missing_count']}\n\n"
        
        numerical_cols = [col for col, info in self.data_dict.items() 
                         if info['type'] == 'numerical' and not info.get('is_target', False)]
        if numerical_cols:
            report += "## Numerical Variables\n\n"
            for col in numerical_cols:
                info = self.data_dict[col]
                report += f"### {col}\n"
                report += f"- Range: {info['min']} - {info['max']}\n"
                report += f"- Mean: {info['mean']:.2f}\n"
                report += f"- Median: {info['median']:.2f}\n"
                report += f"- Missing values: {info['missing_count']}\n\n"
        
        return report

    def plot_feature_distributions(self, df: pd.DataFrame, max_features: int = 20):
        fig = plt.figure(figsize=(15, 10))
        
        numerical_cols = [col for col, info in self.data_dict.items() 
                         if info['type'] == 'numerical' and not info.get('is_target', False)]
        categorical_cols = [col for col, info in self.data_dict.items() 
                           if info['type'] in ['categorical', 'binary'] and not info.get('is_target', False)]
        
        # Limit number of features to plot
        numerical_cols = numerical_cols[:min(len(numerical_cols), max_features//2)]
        categorical_cols = categorical_cols[:min(len(categorical_cols), max_features//2)]
        
        # Plot numerical distributions
        for i, col in enumerate(numerical_cols):
            plt.subplot(len(numerical_cols), 2, 2*i+1)
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f"{col} Distribution")
            
            plt.subplot(len(numerical_cols), 2, 2*i+2)
            sns.boxplot(x=df[col].dropna())
            plt.title(f"{col} Boxplot")
        
        # Plot categorical distributions
        for i, col in enumerate(categorical_cols):
            if i >= max_features//2:
                break
                
            plt.figure(figsize=(10, 5))
            counts = df[col].value_counts().sort_values(ascending=False)
            
            # Limit categories shown if too many
            if len(counts) > 15:
                other_count = counts[15:].sum()
                counts = counts[:15]
                counts['Other'] = other_count
                
            sns.barplot(x=counts.index, y=counts.values)
            plt.title(f"{col} Distribution")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
        
        return plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a dataset for machine learning.')
    parser.add_argument('file', type=str, help='Path to the CSV file')
    parser.add_argument('--name', type=str, default='dataset', help='Name of the dataset')
    parser.add_argument('--target', type=str, help='Target column name')
    parser.add_argument('--id-cols', type=str, help='Comma-separated list of ID columns')
    parser.add_argument('--categorical', type=str, help='Comma-separated list of categorical columns')
    parser.add_argument('--binary', type=str, help='Comma-separated list of binary columns')
    parser.add_argument('--numerical', type=str, help='Comma-separated list of numerical columns')
    parser.add_argument('--output', type=str, default='processed_data.csv', help='Output file path')
    parser.add_argument('--report', type=str, help='Path to save the report')
    parser.add_argument('--no-scale', action='store_true', help='Do not scale numerical features')
    parser.add_argument('--missing', choices=['drop', 'impute'], default='drop', help='How to handle missing values')
    
    args = parser.parse_args()
    
    # Load data
    df = pd.read_csv(args.file)
    
    # Parse column lists
    id_cols = args.id_cols.split(',') if args.id_cols else None
    categorical_cols = args.categorical.split(',') if args.categorical else None
    binary_cols = args.binary.split(',') if args.binary else None
    numerical_cols = args.numerical.split(',') if args.numerical else None
    
    # Process data
    processor = DataProcessor(name=args.name)
    processed_df = processor.process_dataset(
        df=df,
        categorical_cols=categorical_cols,
        binary_cols=binary_cols,
        numerical_cols=numerical_cols,
        target_col=args.target,
        id_cols=id_cols,
        scale_numerical=not args.no_scale,
        handle_missing=args.missing
    )
    
    # Save processed data
    processed_df.to_csv(args.output, index=False)
    
    # Save documentation
    processor.save_documentation()
    
    # Generate and save report if requested
    if args.report:
        report = processor.generate_report()
        with open(args.report, 'w') as f:
            f.write(report)
