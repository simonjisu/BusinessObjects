SELECT SUM(l_extendedprice * (1 - l_discount)) * 100.0 / (SELECT SUM(l_extendedprice * (1 - l_discount)) FROM lineitem l2 JOIN partsupp ps2 ON l2.l_partkey = ps2.ps_partkey JOIN part p2 ON ps2.ps_partkey = p2.p_partkey JOIN orders o2 ON l2.l_orderkey = o2.o_orderkey WHERE o2.o_orderdate BETWEEN '1993-09-01' AND '1993-09-30') AS revenue_percentage
FROM lineitem l
JOIN partsupp ps ON l.l_partkey = ps.ps_partkey
JOIN part p ON ps.ps_partkey = p.p_partkey
JOIN orders o ON l.l_orderkey = o.o_orderkey
WHERE p.p_type LIKE '%PROMO%' AND o.o_orderdate BETWEEN '1993-09-01' AND '1993-09-30';