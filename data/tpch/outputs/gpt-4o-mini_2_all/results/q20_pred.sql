SELECT COUNT(DISTINCT s.s_suppkey) AS supplier_count
FROM supplier s
JOIN nation n ON s.s_nationkey = n.n_nationkey
JOIN partsupp ps ON s.s_suppkey = ps.ps_suppkey
JOIN lineitem l ON ps.ps_partkey = l.l_partkey AND ps.ps_suppkey = l.l_suppkey
JOIN orders o ON l.l_orderkey = o.o_orderkey
WHERE n.n_name = 'CANADA'
AND l.l_shipdate BETWEEN '1993-01-01' AND '1993-12-31'
AND l.l_quantity > 0.5 * ps.ps_availqty;