# ORDER_PRIORITY_CHECKING_3MONTHS
# requires :1 date
select
	o_orderpriority,
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
	)
group by
	o_orderpriority
order by
	o_orderpriority;