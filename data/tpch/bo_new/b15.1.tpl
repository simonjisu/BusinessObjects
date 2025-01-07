# TOP_REVENUE_SUPPLIER
# requires :1 date
select
	s_suppkey, s_name, s_address, s_phone, total_revenue
from
	supplier,
	SUPPLIER_TOTAL_REVENUE(:1) as revenue0
where
	s_suppkey = supplier_no
	and total_revenue = (
		select
			max(total_revenue)
		from
			revenue0
	)
order by
	s_suppkey