select
	l_returnflag,
	l_linestatus,
	sum(l_extendedprice) as sum_price,
	sum(l_extendedprice * (1 - l_discount)) as sum_disc_price,
	sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge,
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