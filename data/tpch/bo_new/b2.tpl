# MINIMUM_COST_SUPPLYCOST
# requires :1 r_name
select
		ps_partkey, min(ps_supplycost) as min_ps_supplycost
from
	partsupp,
	supplier,
	nation,
	region
where
	s_suppkey = ps_suppkey
	and s_nationkey = n_nationkey
	and n_regionkey = r_regionkey
	and r_name = ':1'
group by ps_partkey