SELECT COUNT(DISTINCT T1.PS_SUPPKEY) FROM PARTSUPP T1 JOIN PART T2 ON T1.PS_PARTKEY = T2.P_PARTKEY WHERE T2.P_SIZE = 49