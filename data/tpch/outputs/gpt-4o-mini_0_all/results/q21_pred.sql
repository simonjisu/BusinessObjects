SELECT S.S_SUPPKEY, COUNT(*) AS late_delivery_count
FROM SUPPLIER S
JOIN LINEITEM L ON S.S_SUPPKEY = L.L_SUPPKEY
WHERE S.S_NATIONKEY = (SELECT N.N_NATIONKEY FROM NATION N WHERE N.N_NAME = 'GERMANY')
AND L.L_COMMITDATE < L.L_RECEIPTDATE
GROUP BY S.S_SUPPKEY
ORDER BY late_delivery_count DESC
LIMIT 100;