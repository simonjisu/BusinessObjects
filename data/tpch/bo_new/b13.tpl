# SPECIAL_CATEGORY_CUSTOMERS_ORDERS
# requires :1 comment
select
	c_custkey,
	count(o_orderkey)
from
	customer left outer join orders on
	c_custkey = o_custkey
	and (o_comment not like '%:1%')
group by
	c_custkey