# SMALL_QUANTITY_ORDER
# requires :1 p_brand, :2 p_container, :3 fraction
select
	sum(l_extendedprice) / (datediff('years', min(o_orderdate), max(o_orderdate))) as avg_yearly
from
	lineitem,
	part,
	orders,
	(PARTIAL_AVERAGE_QUANTITY(:3)) part_agg
where
	p_partkey = l_partkey
	and agg_partkey = l_partkey
	and l_quantity < avg_quantity
	and p_brand = ':1'
	and p_container = ':2';