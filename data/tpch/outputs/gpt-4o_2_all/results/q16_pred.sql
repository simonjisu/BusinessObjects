SELECT COUNT(DISTINCT S_SUPPKEY) AS supplier_count
FROM PART AS T1
JOIN PARTSUPP AS T2 ON T1.P_PARTKEY = T2.PS_PARTKEY
JOIN SUPPLIER AS T3 ON T2.PS_SUPPKEY = T3.S_SUPPKEY
WHERE T1.P_SIZE = 49;