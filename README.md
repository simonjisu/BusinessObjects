# BusinessLake

Business Lake Product

setup

* create the folder `.gcp` and put the BigQuery key file in it.
* create `.env` file and put the `OPENAI_API_KEY`, `GCP_PROJECT_ID` in it.
* after install postgresql

```bash
$ sudo service postgresql start
$ sudo -u postgres psql
postgres=# CREATE ROLE simonjisu LOGIN SUPERUSER CREATEDB CREATEROLE REPLICATION BYPASSRLS;
postgres=# exit
$ psql postgres simonjisu
postgres=# CREATE DATABASE retail
    WITH 
    OWNER = simonjisu
    ENCODING = 'UTF8'
    CONNECTION LIMIT = -1;
postgres=# exit
$ psql -U simonjisu -d retail -a -f db.sql
```

follow the instruction in GCP Migration from PostgreSQL to BigQuery: [link](https://cloud.google.com/dataflow/docs/guides/templates/provided/postgresql-to-bigquery?hl=ko#console)


* install gcloud sdk: [link](https://cloud.google.com/sdk/docs/install?hl=ko#linux)


```
"duckdb>=1.0.0",
    "ipykernel>=6.29.5",
    "ipywidgets>=8.1.5",
    "langchain-openai>=0.1.23",
    "litellm>=1.44.22",
    "matplotlib>=3.9.2",
    "numpy>=2.1.1",
    "tqdm>=4.66.5",
    "python-dotenv>=1.0.1",
    "pandas>=2.2.2",
    "torch>=2.4.1",
    "transformers>=4.44.2",
    "sqlglot[rs]>=25.20.1",

```