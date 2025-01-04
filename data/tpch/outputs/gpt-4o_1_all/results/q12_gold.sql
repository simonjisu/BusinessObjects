select
	count(*) as count
from
	orders,
	lineitem
where
	o_orderkey = l_orderkey
	and l_commitdate < l_receiptdate
	and l_receiptdate >= '1993-01-01'
	and l_receiptdate < '1994-01-01';