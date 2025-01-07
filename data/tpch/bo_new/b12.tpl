# SHIPPING_MODES_LATE_ITEMS
# requires :1 list of shipmodes, :2 date
select
	count(o_orderkey)
from
	orders,
	lineitem
where
	o_orderkey = l_orderkey
	and l_commitdate < l_receiptdate
	and l_shipdate < l_commitdate
	and l_shipmode in (:1)
	and l_receiptdate >= date ':2'
	and l_receiptdate < date ':2' + interval '1' year
group by
	l_shipmode
order by
	l_shipmode;