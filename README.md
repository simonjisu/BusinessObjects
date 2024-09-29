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
https://github.com/ygan/Spider-Syn/blob/main/Spider-Syn/dev.json
https://huggingface.co/datasets/aherntech/spider-realistic
```

