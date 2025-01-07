# PART_SUPP_FRACTION_AVG_QUANTITY
# requires :1 level_of_fraction, :2 date
select
	l_partkey agg_partkey,
	l_suppkey agg_suppkey,
	:1 * sum(l_quantity) AS agg_quantity
from
	lineitem
where
	l_shipdate >= ':2'
	and l_shipdate < ':2' + '1 year'
group by
	l_partkey,
	l_suppkey;