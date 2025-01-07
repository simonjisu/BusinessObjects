select
	count(*) as order_count
from
	orders
where
	o_orderdate >= '1993-01-01'
	and o_orderdate < date('1993-01-01', '+3 months')
	and exists (
		select
			*
		from
			lineitem
		where
			l_orderkey = o_orderkey
			and l_commitdate < l_receiptdate
	);