# ORDERDATE_NATION_VOLUME
# requires :1 r_name, :2 p_type, :3 date1, :4 date2
select
	o_orderdate,
	n2.n_name as nation,
	l_extendedprice * (1 - l_discount) as volume
from
	part,
	supplier,
	lineitem,
	orders,
	customer,
	nation n1,
	nation n2,
	region
where
	p_partkey = l_partkey
	and s_suppkey = l_suppkey
	and l_orderkey = o_orderkey
	and o_custkey = c_custkey
	and c_nationkey = n1.n_nationkey
	and n1.n_regionkey = r_regionkey
	and s_nationkey = n2.n_nationkey
	and r_name = ':1'
	and p_type = ':2'
	and o_orderdate between ':3' and ':4';