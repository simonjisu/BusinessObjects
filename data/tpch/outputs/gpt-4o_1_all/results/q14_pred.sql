SELECT SUM(L.L_EXTENDEDPRICE * (1 - L.L_DISCOUNT)) AS promo_revenue, 
       (SELECT SUM(L2.L_EXTENDEDPRICE * (1 - L2.L_DISCOUNT)) 
        FROM LINEITEM L2 
        WHERE L2.L_SHIPDATE >= '1993-09-01' AND L2.L_SHIPDATE < '1993-10-01') AS total_revenue, 
       (SUM(L.L_EXTENDEDPRICE * (1 - L.L_DISCOUNT)) / 
        (SELECT SUM(L2.L_EXTENDEDPRICE * (1 - L2.L_DISCOUNT)) 
         FROM LINEITEM L2 
         WHERE L2.L_SHIPDATE >= '1993-09-01' AND L2.L_SHIPDATE < '1993-10-01')) * 100 AS promo_revenue_percentage 
FROM LINEITEM L 
JOIN PART P ON L.L_PARTKEY = P.P_PARTKEY 
WHERE P.P_TYPE LIKE '%PROMO%' 
  AND L.L_SHIPDATE >= '1993-09-01' 
  AND L.L_SHIPDATE < '1993-10-01';