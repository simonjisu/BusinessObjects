# PRODUCT_TYPE_PROFIT_MEASURE
# requires :1 p_name
select
	nation,
	extract(year from o_orderdate) as o_year,
	sum(amount) as sum_profit
from
	PRODUCT_TYPE_PROFIT_AMOUNT_BY_ORDERDATE(:1) as profit
group by
	nation,
	o_year
order by
	nation,
	o_year desc;