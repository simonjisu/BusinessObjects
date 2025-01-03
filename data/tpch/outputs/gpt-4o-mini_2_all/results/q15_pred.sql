SELECT s.s_suppkey
FROM supplier s
JOIN lineitem l ON s.s_suppkey = l.l_suppkey
JOIN orders o ON l.l_orderkey = o.o_orderkey
WHERE l.l_shipdate >= '1993-03-01' AND l.l_shipdate < '1993-06-01'
GROUP BY s.s_suppkey
ORDER BY SUM(l.l_extendedprice * (1 - l.l_discount)) DESC
LIMIT 1;