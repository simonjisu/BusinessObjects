from pathlib import Path
import duckdb
import json

# Get answers for each query
def get_answers(db_path):
    if not (db_path / 'answers').exists():
        (db_path / 'answers').mkdir(exist_ok=True)
    with duckdb.connect(str(db_path / 'tpch.db')) as con:
        for sql_p in sorted((db_path / 'queries').glob('*.sql'), key=lambda x: int(x.stem)):
            with sql_p.open('r') as f:
                query = f.read()
                df = con.execute(query).df()

            df.to_csv(db_path / 'answers' / (str(sql_p.name.removesuffix('.sql')) + '.csv'), index=False)

# Get data dictionary
def create_data_dictionary(db_path: Path):
    for raw_dd_p in sorted((db_path / 'data_dictionary' / 'raw').glob('*.json')):
        data_dictionary = {}
        table_name = raw_dd_p.stem
        with raw_dd_p.open('r') as f:
            dd = json.load(f)
        
        for col_name, col_dd in dd.items():
            data_dictionary[col_name.lower()] = col_dd['desc']['text']
        
        with (db_path / 'data_dictionary' / f'{table_name}.json').open('w') as f:
            json.dump(data_dictionary, f, indent=4)

def create_db_and_load_data(db_path: Path):
    with duckdb.connect(str(db_path / 'tpch.db')) as con:

        con.execute('''
        DROP TABLE IF EXISTS lineitem;
        DROP TABLE IF EXISTS orders;
        DROP TABLE IF EXISTS partsupp;               
        DROP TABLE IF EXISTS customer;
        DROP TABLE IF EXISTS supplier; 
        DROP TABLE IF EXISTS nation;
        DROP TABLE IF EXISTS region;
        DROP TABLE IF EXISTS part;
        ''')
        con.commit()

        with (db_path / 'tpch-load.sql').open('r') as file:
            s = file.read()
            s = s.replace('[PATH]', str(proj_path / 'db' / 'tables'))
            con.execute(s)
        print('finished loading tables')

        with (db_path / 'tpch-index.sql').open('r') as file:
            s = file.read()
            con.execute(s)

        print('finished create indexes')

if __name__ == '__main__':
    
    proj_path = Path('.').resolve()
    assert proj_path.name == 'nl2sql', 'Please run this script from the root of the project directory'
    db_path = proj_path / 'db'

    create_db_and_load_data(db_path)
    # get_answers(db_path)
    # create_data_dictionary(db_path)