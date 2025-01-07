# POTENTIAL_PART_PROMOTION
# requires :1 date, :2 p_name, :3 n_name
select
	s_name,
	s_address
from
	supplier,
	nation
where
	s_suppkey in (
		PART_EXCESS_SUPPLIER(0.5, :1, :2)
	)
	and s_nationkey = n_nationkey
	and n_name = ':3'
order by
	s_name;