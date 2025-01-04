SELECT SUM(l.l_extendedprice * l.l_discount) AS total_discounted_revenue
FROM LINEITEM AS l
JOIN ORDERS AS o ON l.l_orderkey = o.o_orderkey
JOIN PART AS p ON l.l_partkey = p.p_partkey
WHERE l.l_shipmode = 'AIR'
AND p.p_brand = 'Brand#12';