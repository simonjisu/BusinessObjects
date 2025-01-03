
with revenue0 as
	(select
		l_suppkey as supplier_no,
		sum(l_extendedprice * (1 - l_discount)) as total_revenue
	from
		lineitem
	where
		l_shipdate >= '1993-03-01'
		and l_shipdate < date('1993-03-01', '+3 months')
	group by
		l_suppkey)

select
	s_suppkey, s_name
from
	supplier,
	revenue0
where
	s_suppkey = supplier_no
	and total_revenue = (
		select
			max(total_revenue)
		from
			revenue0
	)
order by
	total_revenue desc
limit 1;