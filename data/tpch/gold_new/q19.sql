select
	sum(l_extendedprice* (1 - l_discount)) as revenue
from
	lineitem,
	part
where
	(
		p_partkey = l_partkey
        and p_brand = 'Brand#12'
		and l_shipmode in ('AIR', 'AIR REG')
	);