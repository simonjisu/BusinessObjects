SELECT C.C_CUSTKEY
FROM CUSTOMER C
LEFT JOIN ORDERS O ON C.C_CUSTKEY = O.O_CUSTKEY
GROUP BY C.C_CUSTKEY
ORDER BY COUNT(O.O_ORDERKEY) DESC;