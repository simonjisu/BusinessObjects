# FORECASTING_REVENUE_CHANGE
# requires :1 discount, :2 quantity
select
	sum(l_extendedprice * l_discount) as revenue
from
	lineitem
where
	l_discount between (:1)
	and l_quantity < :2;