import pandas as pd
from pathlib import Path
import logging
from typing import Dict
import json

def load_tables_description_json(db_directory_path: str, use_value_description: bool) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Loads table descriptions from json files in the database directory.

    Args:
        db_directory_path (str): The path to the database directory.
        use_value_description (bool): Whether to include value descriptions.

    Returns:
        Dict[str, Dict[str, Dict[str, str]]]: A dictionary containing table descriptions.
    """
    description_path = Path(db_directory_path) / "data_dictionary"
    
    if not description_path.exists():
        logging.warning(f"Description path does not exist: {description_path}")
        return {}
    
    table_description = {}
    for json_file in description_path.glob("*.json"):
        table_name = json_file.stem.lower().strip()
        table_description[table_name] = {}
        could_read = False
        try:
            table_description_json = json.load(open(json_file, 'r'))
            for key, value in table_description_json.items():
                column_name = key
                expanded_column_name = column_name
                column_description = value
                data_format = ""
                value_description = ""

                table_description[table_name][column_name.lower().strip()] = {
                    "original_column_name": column_name,
                    "column_name": expanded_column_name,
                    "column_description": column_description,
                    "data_format": data_format,
                    "value_description": value_description
                }
            logging.info(f"Loaded descriptions from {json_file}")
            could_read = True
        except Exception as e:
            print(e)
        if not could_read:
            logging.warning(f"Could not read descriptions from {json_file}")
    return table_description

def load_tables_concatenated_description(db_directory_path: str, use_value_description: bool) -> Dict[str, Dict[str, str]]:
    """
    Loads concatenated table descriptions from the database directory.

    Args:
        db_directory_path (str): The path to the database directory.
        use_value_description (bool): Whether to include value descriptions.

    Returns:
        Dict[str, Dict[str, str]]: A dictionary containing concatenated table descriptions.
    """
    tables_description = load_tables_description(db_directory_path, use_value_description)
    concatenated_descriptions = {}
    
    for table_name, columns in tables_description.items():
        concatenated_descriptions[table_name] = {}
        
        for column_name, column_info in columns.items():
            concatenated_description = ", ".join(
                value for key, value in column_info.items() if key in ['column_name', 'column_description', 'value_description'] and value
            ).strip().replace("  ", " ")
            concatenated_descriptions[table_name][column_name] = concatenated_description.strip(", ")
    
    return concatenated_descriptions
