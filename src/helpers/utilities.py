import concurrent.futures
from datetime import datetime
import os
from pathlib import Path
from typing import List, Tuple, Union
import pandas as pd
import yaml

class FileHandler:
    ROOT_DIRECTORY = os.path.join(*Path(os.path.dirname(os.path.abspath(__file__))).parts[0:-2])
    SOURCE_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'src')
    SETTINGS_FILEPATH = os.path.join(SOURCE_DIRECTORY, 'settings.yaml')
    DATA_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'data')

    @staticmethod
    def read_yaml(filepath: Union[str, Path]) -> dict:
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)

        return data
    
    @staticmethod
    def get_settings() -> dict:
        return FileHandler.read_yaml(FileHandler.SETTINGS_FILEPATH)
    

class Southern_Company_Smart_Neighborhood:
    DEFAULT_DATA_DIRECTORY = Path(FileHandler.get_settings()['southern_company_smart_neighborhood']['root_directory'])
    METADATA_DIRECTORY = os.path.join(DEFAULT_DATA_DIRECTORY, 'metadata')
    FIGURES_DIRETORY = os.path.join(DEFAULT_DATA_DIRECTORY, 'figures')

    @staticmethod
    def get_all_file_column_summary_statistics(directory: Union[str, Path] = None, timestamp_columns: List[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Returns column summary statistics in all files and an error log of files
        that were read or preprocessed unsuccessfully. Returns all timestamps in UTC."""

        file_summary = Southern_Company_Smart_Neighborhood.get_all_file_metadata(directory)
        paths = file_summary[file_summary['file_extension'].isin(['.csv', '.parquet'])]['relative_path'].tolist()
        statistics_data = []
        error_log = []
        timestamp_columns = [] if timestamp_columns is None else timestamp_columns

        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            results = [executor.submit(Southern_Company_Smart_Neighborhood.get_file_column_summary_statistics, *(p, timestamp_columns)) for p in paths]

            for i, future in enumerate(concurrent.futures.as_completed(results)):
                try:
                    result = future.result()

                    if isinstance(result[1], Exception):
                        error_log.append([*result])

                    else:
                        statistics_data.append(result[0])
                    
                    print(f'\rCompleted {i + 1}/{len(paths)} ({len(error_log)} errors)', end=' '*100)

                except Exception as e:
                    print(e)

        if len(error_log) > 0:
            error_log = pd.DataFrame(error_log, columns=['relative_path', 'error'])
        else:
            error_log = None

        if len(statistics_data) > 0:
            statistics_data = pd.concat(statistics_data, ignore_index=True)
        else:
            statistics_data = None

        return statistics_data, error_log
    
    @staticmethod
    def get_all_file_column_and_row_summary(directory: Union[str, Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Returns summary of column names and their data type in all files as well as the number of 
        rows and columns in the files. An error log is also returned for files that were read is unsuccessfully."""
        
        file_summary = Southern_Company_Smart_Neighborhood.get_all_file_metadata(directory)
        paths = file_summary[file_summary['file_extension'].isin(['.csv', '.parquet'])]['relative_path'].tolist()
        column_metadata = []
        row_count_metadata = []
        error_log = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            results = [executor.submit(Southern_Company_Smart_Neighborhood.get_file_column_and_row_summary, *(p,)) for p in paths]

            for i, future in enumerate(concurrent.futures.as_completed(results)):
                try:
                    result = future.result()

                    if isinstance(result[1], Exception):
                        error_log.append([*result])

                    else:
                        column_metadata.append(result[0])
                        row_count_metadata.append([result[1], result[2], result[3]])
                    
                    print(f'\rCompleted {i + 1}/{len(paths)} ({len(error_log)} errors)', end=' '*100)

                except Exception as e:
                    print(e)

        if len(error_log) > 0:
            error_log = pd.DataFrame(error_log, columns=['relative_path', 'error'])
        else:
            error_log = None

        if len(column_metadata) > 0:
            column_metadata = pd.concat(column_metadata, ignore_index=True)
        else:
            column_metadata = None

        if len(row_count_metadata) > 0:
            row_count_metadata = pd.DataFrame(row_count_metadata, columns=['relative_path', 'row_count', 'column_count'])
        else:
            row_count_metadata = None

        return column_metadata, row_count_metadata, error_log
    

    @staticmethod
    def get_file_column_summary_statistics(relative_path: Union[Path, str], timestamp_columns: List[str]) -> Union[Tuple[Path, Exception], Tuple[pd.DataFrame, Path]]:
        """Returns column summary statistics in a file. An error log is returned if file 
        read or preprocessing is unsuccessful. Returns all timestamps in UTC."""

        resolutions = {}

        try:
            f = os.path.join(Southern_Company_Smart_Neighborhood.DEFAULT_DATA_DIRECTORY, relative_path)
            file_extension = os.path.splitext(f)[1]
            data = Southern_Company_Smart_Neighborhood.get_file_reader(file_extension)(f)
            
            for c in timestamp_columns:
                if c in data.columns:
                    data[c] = pd.to_datetime(data[c], utc=True)
                    resolutions[c] = pd.Series(data[c].unique()).sort_values().diff()
                
                else:
                    pass
                    
            statistics_data = data.describe(include='all', datetime_is_numeric=True).T
            statistics_data.index.name = 'column_name'
            statistics_data = statistics_data.reset_index()
            statistics_data['categories'] = statistics_data['column_name'].map(lambda x: data[x].unique() if pd.api.types.is_object_dtype(data[x]) else None)
            statistics_data['min_resolution'] = statistics_data['column_name'].map(lambda x: resolutions[x].min().total_seconds() if x in timestamp_columns else None)
            statistics_data['max_resolution'] = statistics_data['column_name'].map(lambda x: resolutions[x].max().total_seconds() if x in timestamp_columns else None)
            statistics_data['mean_resolution'] = statistics_data['column_name'].map(lambda x: resolutions[x].mean().total_seconds() if x in timestamp_columns else None)
            statistics_data.insert(0, 'relative_path', relative_path)
        
        except Exception as e:
            return relative_path, e
        
        return statistics_data, relative_path
    
    @staticmethod
    def get_file_column_and_row_summary(relative_path: Union[Path, str]) -> Union[Tuple[Path, Exception], Tuple[pd.DataFrame, Path, int, int]]:
        """Returns summary of column names and their data type in a file as well as the number of 
        rows and columns in the file. An error log is returned if file read is unsuccessful."""

        try:
            f = os.path.join(Southern_Company_Smart_Neighborhood.DEFAULT_DATA_DIRECTORY, relative_path)
            file_extension = os.path.splitext(f)[1]
            data = Southern_Company_Smart_Neighborhood.get_file_reader(file_extension)(f)
        
        except Exception as e:
            return relative_path, e
        
        column_metadata = data.dtypes.to_frame().reset_index()
        column_metadata.columns = ['column_name', 'dtype']
        column_metadata.insert(0, 'relative_path', relative_path)
        
        return column_metadata, relative_path, data.shape[0], data.shape[1]
    
    @staticmethod
    def get_file_reader(file_extension: str) -> type(pd.read_csv):
        """Return pandas function used to read files with provided file extension."""

        return {'.csv': pd.read_csv, '.parquet': pd.read_parquet}[file_extension]
    
    @staticmethod
    def get_all_file_metadata(directory: Union[str, Path] = None) -> pd.DataFrame:
        """Returns all file metadata include directory, path, extension, size, last access timestamp, last modified timestamp, and create timestamp."""

        directory = Southern_Company_Smart_Neighborhood.DEFAULT_DATA_DIRECTORY if directory is None else directory
        files = [os.path.join(p, n) for p, _, f in os.walk(directory) for n in f]

        data = pd.DataFrame({'full_path': files})
        data['is_directory'] = data['full_path'].map(lambda x: os.path.isdir(x))
        data['relative_path'] = data['full_path'].map(lambda x: Path(x).relative_to(Path(directory)))
        data['relative_directory'] = data['relative_path'].map(lambda x: str(Path(x).parent))
        data['filename'] = data['full_path'].map(lambda x: os.path.split(x)[1])
        data['file_extension'] = data['filename'].map(lambda x: os.path.splitext(x)[1])
        data['file_size'] = data['full_path'].map(lambda x: os.stat(x).st_size)
        data['last_access_timestamp'] = data['full_path'].map(lambda x: datetime.utcfromtimestamp(os.stat(x).st_atime))
        data['last_modified_timestamp'] = data['full_path'].map(lambda x: datetime.utcfromtimestamp(os.stat(x).st_mtime))
        data['create_timestamp'] = data['full_path'].map(lambda x: datetime.utcfromtimestamp(os.stat(x).st_ctime))
        data = data[(data['is_directory']==False) & (data['file_extension']!='')].copy()
        data = data.drop(columns=['full_path', 'is_directory'])
        
        return data