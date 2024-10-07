# FORECASTING_REVENUE_CHANGE_1YEAR
# requires :1 date, :2 discount, :3 quantity
select
	sum(l_extendedprice * l_discount) as revenue
from
	lineitem
where
	l_shipdate >= ':1'
	and l_shipdate < ':1' + '1 year'
	and l_discount between (:2)
	and l_quantity < :3;