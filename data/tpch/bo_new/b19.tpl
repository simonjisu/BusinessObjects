# DISCOUNTED_REVENUE
# requires :1 p_brand, :2 p_container, :3 l_quantity, :4 p_size
select
	sum(l_extendedprice * (1 - l_discount)) as revenue
from
	lineitem,
	part
where
	(
		p_partkey = l_partkey
		and p_brand = ':1'
		and p_container like (:2%)
		and l_quantity >= :3 and l_quantity <= :3
		and p_size between 1 and :4
		and l_shipmode in ('AIR', 'AIR REG')
		and l_shipinstruct = 'DELIVER IN PERSON'
	)