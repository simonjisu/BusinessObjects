# UNSHIPPED_ORDERS_10_HIGHEST_VALUE
# requires :1 c_mktsegment, :2 date
select
	l_orderkey,
	sum(l_extendedprice * (1 - l_discount)) as revenue,
	o_orderdate,
	o_shippriority
from
	customer,
	orders,
	lineitem
where
	c_custkey = o_custkey
	and l_orderkey = o_orderkey
	and o_orderdate < ':2'
	and l_shipdate > ':2'
	and c_mktsegment = ':1'
group by
	l_orderkey,
	o_orderdate,
	o_shippriority
order by
	revenue desc,
	o_orderdate
LIMIT 10;