select
	l_returnflag,
	l_linestatus,
	avg(l_quantity) as avg_qty,
	avg(l_extendedprice) as avg_price,
	avg(l_discount) as avg_disc,
	count(*) as count_order
from
	lineitem
where
	l_shipdate <= date((select max(l_shipdate) from lineitem), '-3 months')
group by
	l_returnflag,
	l_linestatus
order by
	l_returnflag,
	l_linestatus;