# train

db_id: shooting
sample id: 2462
exp: no column called `officer_count` in `incidents`

```json
{
    "db_id": "shooting",
    "question": "What is the percentage of the cases involved more than 3 officers from year 2010 to 2015?",
    "evidence": "more than 3 officers refers to officer_count > 3; from year 2010 to 2015 refers to date between '2010-01-01' and '2015-12-31'; percentage = divide(count(case_number where officer_count > 3), count(case_number)) where date between '2010-01-01' and '2015-12-31' * 100%",
    "SQL": "SELECT CAST(SUM(IIF(officer_count > 3, 1, 0)) AS REAL) * 100 / COUNT(case_number) FROM incidents WHERE STRFTIME('%Y', date) BETWEEN '2010' AND '2015'"
},
```


db_id: shooting
sample id: 2464
exp: no column called `grand_jury_disposition` in `incidents`

```json
{
    "db_id": "shooting",
    "question": "Among the cases dismissed by the grand jury disposition, what percentage of cases is where the subject is injured?",
    "evidence": "dismissed by the grand jury disposition refers to grand_jury_disposition = 'No Bill'; the subject is injured refers to subject_statuses = 'injured'; percentage = divide(count(incidents where subject_statuses = 'injured'), count(incidents)) where grand_jury_disposition = 'No Bill' * 100%",
    "SQL": "SELECT CAST(SUM(subject_statuses = 'Injured') AS REAL) * 100 / COUNT(case_number) FROM incidents WHERE grand_jury_disposition = 'No Bill'"
},
```