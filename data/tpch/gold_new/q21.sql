select
	s_name,
	count(*) as numwait
from
	supplier,
	lineitem l1,
	orders,
	nation
where
	s_suppkey = l1.l_suppkey
	and o_orderkey = l1.l_orderkey
	and l1.l_receiptdate > l1.l_commitdate
	and s_nationkey = n_nationkey
	and n_name = 'GERMANY'
group by
	s_name
order by 
    numwait desc
limit 100;