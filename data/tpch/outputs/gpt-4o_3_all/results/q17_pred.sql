SELECT strftime('%Y', L.L_SHIPDATE) AS year, SUM(L.L_EXTENDEDPRICE) AS gross_revenue
FROM LINEITEM L
JOIN PART P ON L.L_PARTKEY = P.P_PARTKEY
WHERE P.P_BRAND = 'Brand#23'
GROUP BY year;