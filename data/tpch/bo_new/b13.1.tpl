# CUSTOMER_ORDERS_DISTRIBUTION
# requires :1 comment
select
	c_count,
	count(*) as custdist
from
	SPECIAL_CATEGORY_CUSTOMERS_ORDERS(:1) as c_orders (c_custkey, c_count)
group by
	c_count
order by
	c_count, 
	custdist desc;