# AVAIL_SUPPLY_COST
# requires :1 n_name, :2 fraction
select
	sum(ps_supplycost * ps_availqty) * :2
from
	partsupp,
	supplier,
	nation
where
	ps_suppkey = s_suppkey
	and s_nationkey = n_nationkey
	and n_name = ':1'
)