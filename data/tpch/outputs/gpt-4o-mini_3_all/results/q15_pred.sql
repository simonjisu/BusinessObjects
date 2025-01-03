select s.s_suppkey, sum(l.l_extendedprice * (1 - l.l_discount)) as revenue
from supplier s,
     lineitem l,
     orders o
where s.s_suppkey = l.l_suppkey
  and o.o_orderkey = l.l_orderkey
  and o.o_orderdate >= '1993-03-01'
  and o.o_orderdate < '1993-06-01'
group by s.s_suppkey
order by revenue desc
LIMIT 1;