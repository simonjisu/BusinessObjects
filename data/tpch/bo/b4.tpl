# LATE_DELIVERY_ORDERS
# requires :1 date
select
	count(*) as order_count
from
	orders
where
	o_orderdate >= ':1'
	and o_orderdate < ':1' + '3 months'
	and exists (
		select
			*
		from
			lineitem
		where
			l_orderkey = o_orderkey
			and l_commitdate < l_receiptdate
	);