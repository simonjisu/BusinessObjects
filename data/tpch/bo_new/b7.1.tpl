# SUPP_CUST_VOLUME_SHIPPING_BY_YEAR
# requires :1 n_name1, :2 n_name2, :3 date1, :4 date2
select
	supp_nation,
	cust_nation,
	extract(year from l_shipdate) as l_year,
	sum(volume) as revenue
from
	SUPP_CUST_VOLUME_SHIPPING(:1, :2, :3, :4) as shipping
group by
	supp_nation,
	cust_nation,
	l_year
order by
	supp_nation,
	cust_nation,
	l_year;