"""
File handling utilities for Excel and CSV files
"""

import pandas as pd
import os
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def read_input_file(file_path: str, chunksize: Optional[int] = None) -> pd.DataFrame:
    """
    Read Excel or CSV file
    
    Args:
        file_path: Path to input file
        chunksize: If specified, read in chunks (for large files)
        
    Returns:
        Pandas DataFrame
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_extension in ['.xlsx', '.xls']:
            # read Excel file
            if chunksize:
                # excel doesn't support chunksize, read all at once
                logger.warning("Chunksize not supported for Excel files, reading entire file")
            df = pd.read_excel(file_path, engine='openpyxl' if file_extension == '.xlsx' else None)
            
        elif file_extension == '.csv':
            # read CSV file
            if chunksize:
                # return iterator for chunked reading
                return pd.read_csv(file_path, chunksize=chunksize)
            df = pd.read_csv(file_path)
            
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        logger.info(f"Successfully read file: {file_path} ({len(df)} rows)")
        return df
        
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        raise


def write_output_file(df: pd.DataFrame, output_path: str, original_filename: str):
    """
    Write DataFrame to CSV file (always CSV format)
    
    Args:
        df: DataFrame to write
        output_path: Path where file should be saved
        original_filename: Original input filename (for reference)
    """
    try:
        # always write as CSV file
        # change output path extension to .csv
        output_path = output_path.replace('.xlsx', '.csv').replace('.xls', '.csv')
        df.to_csv(output_path, index=False)
        
        logger.info(f"Successfully wrote file: {output_path} ({len(df)} rows)")
        
    except Exception as e:
        logger.error(f"Error writing file {output_path}: {str(e)}")
        raise


def process_large_file(file_path: str, process_func, batch_size: int = 1000) -> pd.DataFrame:
    """
    Process large files in chunks to avoid memory issues
    
    Args:
        file_path: Path to input file
        process_func: Function to apply to each chunk (receives DataFrame, returns DataFrame)
        batch_size: Number of rows per chunk
        
    Returns:
        Combined processed DataFrame
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.csv':
        # process CSV in chunks
        chunks = []
        for chunk in pd.read_csv(file_path, chunksize=batch_size):
            processed_chunk = process_func(chunk)
            chunks.append(processed_chunk)
        
        return pd.concat(chunks, ignore_index=True)
    
    else:
        # for excel, read all at once (openpyxl doesn't support chunking)
        df = pd.read_excel(file_path)
        return process_func(df)


def validate_file_structure(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Validate that DataFrame has all required columns
    
    Args:
        df: Input DataFrame
        required_columns: List of required column names
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    # normalize column names (strip whitespace, lowercase)
    df_columns = [col.strip().lower() for col in df.columns]
    required_columns_normalized = [col.strip().lower() for col in required_columns]
    
    missing_columns = set(required_columns_normalized) - set(df_columns)
    
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            f"Required columns are: {required_columns}"
        )
    
    return True


def get_file_info(file_path: str) -> dict:
    """
    Get information about a file
    
    Args:
        file_path: Path to file
        
    Returns:
        Dictionary with file information
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_size = os.path.getsize(file_path)
    file_extension = os.path.splitext(file_path)[1]
    
    # try to read first few rows to get column info
    try:
        if file_extension.lower() in ['.xlsx', '.xls']:
            df_sample = pd.read_excel(file_path, nrows=5)
        elif file_extension.lower() == '.csv':
            df_sample = pd.read_csv(file_path, nrows=5)
        else:
            df_sample = None
        
        if df_sample is not None:
            return {
                "file_name": os.path.basename(file_path),
                "file_size_bytes": file_size,
                "file_size_mb": round(file_size / (1024 * 1024), 2),
                "file_extension": file_extension,
                "columns": df_sample.columns.tolist(),
                "num_columns": len(df_sample.columns)
            }
    except Exception as e:
        logger.warning(f"Could not read file for info: {str(e)}")
    
    return {
        "file_name": os.path.basename(file_path),
        "file_size_bytes": file_size,
        "file_size_mb": round(file_size / (1024 * 1024), 2),
        "file_extension": file_extension
    }