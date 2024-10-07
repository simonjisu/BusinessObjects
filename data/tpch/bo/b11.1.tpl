# IMPORTANT_STOCK_IDENTIFICATION
# requires :1 n_name, :2 fraction
select
	ps_partkey,
	sum(ps_supplycost * ps_availqty) as value
from
	partsupp,
	supplier,
	nation
where
	ps_suppkey = s_suppkey
	and s_nationkey = n_nationkey
	and n_name = ':1'
group by
	ps_partkey having
		sum(ps_supplycost * ps_availqty) > AVAIL_SUPPLY_COST(:1, :2)
order by
	value desc;