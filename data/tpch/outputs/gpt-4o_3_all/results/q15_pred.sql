SELECT T2.S_SUPPKEY, SUM(T1.L_EXTENDEDPRICE * (1 - T1.L_DISCOUNT)) AS total_revenue
FROM LINEITEM T1
JOIN SUPPLIER T2 ON T1.L_SUPPKEY = T2.S_SUPPKEY
WHERE T1.L_SHIPDATE >= '1993-03-01' AND T1.L_SHIPDATE < '1993-06-01'
GROUP BY T2.S_SUPPKEY
ORDER BY total_revenue DESC
LIMIT 1;