# NATION_MKTSHARE_BY_ORDERYEAR_FOR_REGION
# requires :1 r_name, :2 p_type, :3 date1, :4 date2
select
	extract(year from o_orderdate) as o_year,
    nation,
    sum(volume) as revenue,
    sum(volume) / sum(sum(volume)) OVER (PARTITION BY o_year) as mkt_share
from
	ORDERDATE_NATION_VOLUME(:1, :2, :3, :4) as all_nations
group by
	o_year
order by
	o_year;