data_mode='dev' # Options: 'dev', 'train' 

data_path="./data/dev/test_sample.json" # UPDATE THIS WITH THE PATH TO THE TARGET DATASET

pipeline_nodes='keyword_extraction+entity_retrieval'

# Nodes:
    # keyword_extraction
    # entity_retrieval
    # context_retrieval
    # column_filtering
    # table_selection
    # column_selection
    # candidate_generation
    # revision
    # evaluation


entity_retieval_mode='ask_model' # Options: 'corrects', 'ask_model'

context_retrieval_mode='vector_db' # Options: 'corrects', 'vector_db'
top_k=5

table_selection_mode='ask_model' # Options: 'corrects', 'ask_model'

column_selection_mode='ask_model' # Options: 'corrects', 'ask_model'

engine1='gpt-4o-mini'
engine2='gpt-4o'
engine3='o1-mini'
engine4='o1-preview'
engine5='gemini-1.5-pro'
engine6='gemini-1.5-flash'


pipeline_setup='{
    "keyword_extraction": {
        "engine": "'${engine1}'",
        "temperature": 0.2,
        "base_uri": ""
    },
    "entity_retrieval": {
        "mode": "'${entity_retieval_mode}'"
    }
}'

python3 -u ./src/main.py --data_mode ${data_mode} --data_path ${data_path}\
        --pipeline_nodes ${pipeline_nodes} --pipeline_setup "$pipeline_setup"\
        --engine ${engine1}  
