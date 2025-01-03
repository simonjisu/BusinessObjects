select
	sum(l_extendedprice) / (strftime('%Y', max(o_orderdate))-strftime('%Y', min(o_orderdate))) as avg_yearly
from
	lineitem,
    orders,                                                                            
	part
where
	p_partkey = l_partkey
    and l_orderkey = o_orderkey
	and p_brand = 'Brand#23';