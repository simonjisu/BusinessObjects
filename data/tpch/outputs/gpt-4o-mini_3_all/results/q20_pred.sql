SELECT COUNT(DISTINCT s.s_suppkey) AS supplier_count
FROM partsupp ps
JOIN supplier s ON ps.ps_suppkey = s.s_suppkey
JOIN nation n ON s.s_nationkey = n.n_nationkey
JOIN lineitem l ON ps.ps_partkey = l.l_partkey
WHERE n.n_name = 'CANADA'
AND l.l_shipdate BETWEEN '1993-01-01' AND '1993-12-31'
AND l.l_quantity > 0.5 * ps.ps_availqty;