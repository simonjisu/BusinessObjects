select
	sum(l_extendedprice) / (strftime('%Y', max(o_orderdate))-strftime('%Y', min(o_orderdate))) as avg_yearly
from
	lineitem,
    orders,                                                                            
	part,
	(SELECT 
		l_partkey AS agg_partkey, 
		0.2 * avg(l_quantity) AS avg_quantity 
	FROM lineitem 
	GROUP BY l_partkey) part_agg
where
	p_partkey = l_partkey
    and l_orderkey = o_orderkey
	and agg_partkey = l_partkey
	and p_brand = 'Brand#23'
	and p_container = 'MED BOX'
	and l_quantity < avg_quantity;