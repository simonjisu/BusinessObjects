# PARTIAL_AVERAGE_QUANTITY
# requires :1 fraction
SELECT 
	l_partkey AS agg_partkey, 
	:1 * avg(l_quantity) AS avg_quantity 
FROM lineitem 
GROUP BY l_partkey