SELECT COUNT(DISTINCT T1.O_CUSTKEY) AS num_customers
FROM ORDERS AS T1
JOIN (
    SELECT L_ORDERKEY
    FROM LINEITEM
    GROUP BY L_ORDERKEY
    HAVING SUM(L_QUANTITY) > 300
) AS T2 ON T1.O_ORDERKEY = T2.L_ORDERKEY;