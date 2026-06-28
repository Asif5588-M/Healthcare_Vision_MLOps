import os
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi

# Configured to load custom secret variables securely
load_dotenv(dotenv_path="config.env")

class DataIngestion:
    def __init__(self, download_path="data/"):
        self.download_path = download_path

    def initiate_data_ingestion(self, dataset_slug):
        """
        Automated script to handle data download streaming workflows.
        """
        try:
            if not os.getenv("KAGGLE_USERNAME") or not os.getenv("KAGGLE_KEY"):
                raise ValueError("Kaggle credentials missing from configuration file!")

            print("Initializing Kaggle API authentication...")
            api = KaggleApi()
            api.authenticate()
            
            os.makedirs(self.download_path, exist_ok=True)
            print(f"Downloading target dataset artifact stream: '{dataset_slug}'...")
            
            # Downloading zipped file directories natively
            api.dataset_download_files(dataset_slug, path=self.download_path, unzip=True)
            print(f"Dataset successfully downloaded and extracted into target path: '{self.download_path}'")
            
        except Exception as e:
            print(f"Error encountered during dataset ingestion phase: {e}")
            raise e
