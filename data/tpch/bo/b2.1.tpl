# MINIMUM_COST_SUPPLIER_TOP100
# requires :1 p_size, :2 p_type, :3 r_name
select
	s_acctbal, s_name, n_name, p_partkey, p_mfgr, s_address, s_phone, s_comment
from
	part, supplier, partsupp, nation, region
where
	ps_partkey = p_partkey
	and p_size = :1
	and p_type like '%:2'
	and s_suppkey = ps_suppkey
	and s_nationkey = n_nationkey
	and n_regionkey = r_regionkey
	and r_name = ':3'
	and ps_supplycost = (select min_ps_supplycost from MINIMUM_COST_SUPPLYCOST(':3') where ps_partkey = p_partkey)
order by
	s_acctbal desc,
	n_name,
	s_name,
	p_partkey
LIMIT 100;