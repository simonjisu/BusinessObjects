# SUPPLIER_TOTAL_REVENUE
# requires :1 date
select
	l_suppkey as supplier_no,
	sum(l_extendedprice * (1 - l_discount)) as total_revenue
from
	lineitem
where
	l_shipdate >= ':1'
	and l_shipdate < ':1' + '3 months'
group by
	l_suppkey;