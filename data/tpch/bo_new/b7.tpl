# SUPP_CUST_SHIPPING_VOLUME
# requires :1 n_name1, :2 n_name2, :3 date1, :4 date2
select
	n1.n_name as supp_nation,
	n2.n_name as cust_nation,
	l_shipdate as l_shipdate,
	l_extendedprice * (1 - l_discount) as volume
from
	supplier,
	lineitem,
	orders,
	customer,
	nation n1,
	nation n2
where
	s_suppkey = l_suppkey
	and o_orderkey = l_orderkey
	and c_custkey = o_custkey
	and s_nationkey = n1.n_nationkey
	and c_nationkey = n2.n_nationkey
	and (
		(n1.n_name = ':1' and n2.n_name = ':2')
		or (n1.n_name = ':2' and n2.n_name = ':1')
	)
	and l_shipdate between ':3' and ':4'