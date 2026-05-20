import kagglehub
import os
import argparse
import sys

def ingesta_datos(output_path):

    print("Iniciando la ingesta de datos...")
    try:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            print(f"Carpeta de salida creada: {output_path}")

        kagglehub.dataset_download("munaalhawawreh/xiiotid-iiot-intrusion-dataset", output_dir = output_path)
        print("Dataset descargado exitosamente.")
        return True

    except Exception as e:
        print(f"Error en la ingesta del dataset: {e}")
        return False


if __name__ == "__main__":
    app_root = os.getenv("APP_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    raw_data_dir = os.getenv("RAW_DATA_DIR", os.path.join(app_root, "datos", "crudo"))

    parser = argparse.ArgumentParser(description="Ingesta de datos desde Kaggle")
    parser.add_argument(
        "--output_path",
        type=str, 
        default = raw_data_dir, 
        help = "Ruta de salida para los datos descargados"
    )

    args = parser.parse_args()
    if not ingesta_datos(args.output_path):
        sys.exit(1)
