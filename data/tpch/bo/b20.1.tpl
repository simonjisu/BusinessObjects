# PART_EXCESS_SUPPLIER
# requires :1 date, :2 p_name
select
	ps_suppkey
from
	partsupp,
	PART_SUPP_FRACTION_AVG_QUANTITY(0.5, :1) as agg_lineitem
where
	agg_partkey = ps_partkey
	and agg_suppkey = ps_suppkey
	and ps_partkey in (
		select
			p_partkey
		from
			part
		where
			p_name like ':2%'
	)
	and ps_availqty > agg_quantity