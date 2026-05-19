import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler
import gc
from sklearn.decomposition import PCA
from collections import Counter
import argparse
import os
import joblib
import sys
from functools import lru_cache
from drift import build_drift_reference


@lru_cache(maxsize=8)
def cargar_artefactos_inferencia(artefactos_path):
    return joblib.load(os.path.abspath(artefactos_path))


def preprocesamiento_inferencia(X_input, artefactos_path):
    artefactos = cargar_artefactos_inferencia(artefactos_path)

    # Cargar cosas
    cols_cat = artefactos["cols_cat"]
    cols_num = artefactos["cols_num"]
    imputer = artefactos["imputer"]
    scaler = artefactos["scaler"]
    encoder = artefactos["encoder"]
    cols_finales = artefactos["columnas_finales"]

    # Limpieza
    common_replacements = {'-': np.nan, '?': np.nan, 'nan': np.nan, 'NaN': np.nan}
    for col in X_input.select_dtypes(include=['object', 'string']).columns:
        X_input[col] = X_input[col].astype('string').str.lower().replace(common_replacements)

    X_input = fix_dtype(X_input)

    # Numéricas 
    X_input[cols_num] = imputer.transform(X_input[cols_num])
    X_input[cols_num] = scaler.transform(X_input[cols_num])

    # Categóricas
    for col in cols_cat:
        X_input[col] = X_input[col].fillna('missing')

    encoded = encoder.transform(X_input[cols_cat])
    X_input = X_input.drop(columns=cols_cat)

    df_encoded = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(cols_cat),
        index=X_input.index
    )

    X_input = pd.concat([X_input, df_encoded], axis=1)

    # Selección final
    X_input = X_input.reindex(columns=cols_finales, fill_value=0)

    return X_input

def preprocesamiento_inicial(datos_crudos, output_dir):

    # No existe el archivo de datos crudos
    if not os.path.isfile(datos_crudos):
        print(f"Error: El archivo de datos crudos no existe: {datos_crudos}")
        return False

    # Comprobación de la carpeta de salida
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Carpeta de salida creada: {output_dir}")

    print("Iniciando el preprocesamiento de los datos...")
    try:
        # Carga de datos
        df = pd.read_csv(datos_crudos, low_memory=False)

        # Limpieza de caracteres no válidos
        common_replacements = {'-': np.nan, '?': np.nan, 'nan': np.nan, 'NaN': np.nan}
        for col in df.select_dtypes(include=['object', 'string']).columns:
            df[col] = df[col].astype('string').str.lower().replace(common_replacements)

        y = df[['class1', 'class2', 'class3']]
        X = df.drop(columns=['class1', 'class2', 'class3'], errors='ignore')    

        # División 70% / 15% / 15% estratificada
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y['class3'])
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=(0.15/0.85), random_state=42, stratify=y_temp['class3'])
        X_test_api = preparar_datos_api(X_test)

        X_train = fix_dtype(X_train)
        X_val = fix_dtype(X_val)
        X_test = fix_dtype(X_test)

        # Eliminación de variables agnósticas (IPs, puertos, timestamps)
        columnas_agnosticas = [
            'scr_ip', 'scr_port', 'des_ip', 'des_port', 'scr_bytes', 'des_bytes',
            'scr_pkts', 'des_pkts', 'scr_packts_ratio', 'des_pkts_ratio',
            'scr_bytes_ratio', 'des_bytes_ratio', 'scr_ip_bytes', 'des_ip_bytes',
            'timestamp', 'date'
        ]
        columnas_a_borrar = [col for col in X_train.columns if col.lower() in columnas_agnosticas]
        X_train = X_train.drop(columns=columnas_a_borrar)

        numeric_train = X_train.select_dtypes(include=[np.number])
        varianzas = numeric_train.var().sort_values()
        # Eliminar varianza = 0
        cols_var_0 = varianzas[varianzas == 0].index.tolist()
        X_train = X_train.drop(columns=cols_var_0, errors='ignore')
        numeric_train = numeric_train.drop(columns=cols_var_0, errors='ignore')
       
        # --- 2. Correlación con el Target ---
        y_train_bin = y_train['class3'].apply(lambda x: 0 if x == 'normal' else 1)
        corr_with_target = numeric_train.apply(lambda col: col.corr(y_train_bin)).abs().sort_values(ascending=False)
        # Eliminar correlación < 0.025
        cols_baja_corr = corr_with_target[corr_with_target < 0.025].index.tolist()
        X_train = X_train.drop(columns=cols_baja_corr, errors='ignore')
        numeric_train = numeric_train.drop(columns=cols_baja_corr, errors='ignore')

        # --- 3. Clustering jerárquico (Redundancia) ---
        corr_matrix = numeric_train.corr().abs()
        # Eliminar redundantes > 0.97
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        cols_redundantes = [column for column in upper.columns if any(upper[column] > 0.97)]
        X_train = X_train.drop(columns=cols_redundantes, errors='ignore')

        # Fase Final de Preprocesamiento: Imputación, Escalado, Codificación y PCA
        X_val = X_val[X_train.columns]
        X_test = X_test[X_train.columns]

        cols_cat = X_train.select_dtypes(include=['object', 'string']).columns.tolist()
        cols_num_bool = X_train.select_dtypes(exclude=['object', 'string']).columns.tolist()

        X_train[cols_num_bool] = X_train[cols_num_bool].astype(np.float32)

        if cols_num_bool:
            imputer_num = SimpleImputer(strategy='mean')
            X_train[cols_num_bool] = imputer_num.fit_transform(X_train[cols_num_bool])
            X_val[cols_num_bool] = imputer_num.transform(X_val[cols_num_bool].astype(np.float32))
            X_test[cols_num_bool] = imputer_num.transform(X_test[cols_num_bool].astype(np.float32))

        if cols_cat:
            for col in cols_cat:
                X_train[col] = X_train[col].fillna('missing')
                X_val[col] = X_val[col].fillna('missing')
                X_test[col] = X_test[col].fillna('missing')

        if cols_num_bool:
            scaler = RobustScaler()
            X_train[cols_num_bool] = scaler.fit_transform(X_train[cols_num_bool]).astype(np.float32)
            X_val[cols_num_bool] = scaler.transform(X_val[cols_num_bool]).astype(np.float32)
            X_test[cols_num_bool] = scaler.transform(X_test[cols_num_bool]).astype(np.float32)

        if cols_cat:
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float32)

            encoded_train = encoder.fit_transform(X_train[cols_cat])
            X_train.drop(columns=cols_cat, inplace=True)
            X_train = pd.concat([X_train, pd.DataFrame(encoded_train, columns=encoder.get_feature_names_out(cols_cat), index=X_train.index)], axis=1)
            del encoded_train

            encoded_val = encoder.transform(X_val[cols_cat])
            X_val.drop(columns=cols_cat, inplace=True)
            X_val = pd.concat([X_val, pd.DataFrame(encoded_val, columns=encoder.get_feature_names_out(cols_cat), index=X_val.index)], axis=1)
            del encoded_val

            encoded_test = encoder.transform(X_test[cols_cat])
            X_test.drop(columns=cols_cat, inplace=True)
            X_test = pd.concat([X_test, pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(cols_cat), index=X_test.index)], axis=1)
            del encoded_test
            gc.collect()
        
        pca = PCA(n_components=0.95, random_state=42)
        pca.fit(X_train)

        # Obtener nombres de variables originales
        original_feature_names = np.array(X_train.columns)

        # Contador de importancia
        feature_counter = Counter()

        num_top_features = 20

        for comp in pca.components_:
            top_indices = np.argsort(np.abs(comp))[-num_top_features:]
            top_features = original_feature_names[top_indices]
            feature_counter.update(top_features)

        # Variables ordenadas por frecuencia de aparición
        cols_pcafss = [feature for feature, _ in feature_counter.most_common()]

        # Filtrar datasets
        X_train_final = X_train[cols_pcafss].copy()
        X_val_final = X_val[cols_pcafss].copy()
        X_test_final = X_test[cols_pcafss].copy()

        # Exportar datasets preprocesados para X e y
        X_train_final.to_csv(f'{output_dir}/X_train_final.csv', index=False)
        X_val_final.to_csv(f'{output_dir}/X_val_final.csv', index=False)
        X_test_final.to_csv(f'{output_dir}/X_test_final.csv', index=False)
        X_test_api.to_csv(f'{output_dir}/X_test_api.csv', index=False)

        for split_name, y_split in [('train', y_train), ('val', y_val), ('test', y_test)]:
            for cls in ['class1', 'class2', 'class3']:
                y_split[[cls]].to_csv(f'{output_dir}/y_{split_name}_{cls}.csv', index=False)
        
        # Características para inferencia
        artefactos = {
            "columnas_finales": cols_pcafss,
            "columnas_modelo": X_train.columns.tolist(),
            "cols_cat": cols_cat,
            "cols_num": cols_num_bool,
            "imputer": imputer_num,
            "scaler": scaler,
            "encoder": encoder,
            "columnas_agnosticas": columnas_agnosticas,
            "cols_var_0": cols_var_0,
            "cols_baja_corr": cols_baja_corr,
            "cols_redundantes": cols_redundantes
        }

        joblib.dump(artefactos, f"{output_dir}/artefactos_inferencia.joblib")

        normal_mask = y_val["class3"].astype(str).str.lower() == "normal"
        referencia_deriva = X_val_final.loc[normal_mask]
        if referencia_deriva.empty:
            referencia_deriva = X_val_final

        build_drift_reference(
            referencia_deriva,
            f"{output_dir}/drift_reference.json",
            source="validation_normal_class3",
        )

        print("Preprocesamiento completado exitosamente.")
        return True 

    except Exception as e:
        print(f"Error en el preprocesamiento de los datos: {e}")
        return False 

def fix_dtype(df, umbral_numerico=0.7):
    object_cols = df.select_dtypes(include=['object', 'string']).columns
    int_cols = df.select_dtypes(include=['int64']).columns
    bool_cols = df.select_dtypes(include=['bool']).columns

    # Convertir booleanos a float
    df[bool_cols] = df[bool_cols].astype(float)

    for col in object_cols:
        valores_unicos = df[col].dropna().unique()

        if {"true", "false"} <= set(valores_unicos):  # Verifica si ambos existen
            df[col] = df[col].map({'true': 1, 'false': 0}).astype(float)
        else:
            converted = pd.to_numeric(df[col], errors='coerce')
            if converted.notna().mean() > umbral_numerico:
                df[col] = converted.astype(float)

    for col in int_cols:
        df[col] = df[col].astype(float)

    return df


def preparar_datos_api(df):
    df_api = df.copy()
    df_api = df_api.rename(
        columns={
            "FIN or RST": "FIN_or_RST",
            "Avg_num_Proc/s": "Avg_num_Proc_s",
            "Std_num_proc/s": "Std_num_proc_s",
            "Avg_num_cswch/s": "Avg_num_cswch_s",
            "std_num_cswch/s": "std_num_cswch_s",
            "read_write_physical.process": "read_write_physical_process",
        }
    )
    return df_api

if __name__ == "__main__":
    app_root = os.getenv("APP_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    raw_data_dir = os.getenv("RAW_DATA_DIR", os.path.join(app_root, "datos", "crudo"))
    data_dir = os.getenv("DATA_DIR", os.path.join(app_root, "datos", "preprocesados"))

    parser = argparse.ArgumentParser(description="Preprocesamiento de datos")
    parser.add_argument(
        "--datos_crudos",
        type=str, 
        default = os.path.join(raw_data_dir, "X-IIoTID dataset.csv"), 
        help = "Ruta del archivo CSV con los datos crudos"
    )
    parser.add_argument(
        "--output_dir",
        type=str, 
        default = data_dir, 
        help = "Ruta del directorio de salida para los datos preprocesados"
    )

    args = parser.parse_args()
    if not preprocesamiento_inicial(args.datos_crudos, args.output_dir):
        sys.exit(1)
