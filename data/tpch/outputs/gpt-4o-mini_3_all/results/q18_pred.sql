SELECT COUNT(DISTINCT c.C_CUSTKEY) AS customer_count
FROM CUSTOMER c
JOIN ORDERS o ON c.C_CUSTKEY = o.O_CUSTKEY
JOIN LINEITEM l ON o.O_ORDERKEY = l.L_ORDERKEY
WHERE l.L_QUANTITY > 300;