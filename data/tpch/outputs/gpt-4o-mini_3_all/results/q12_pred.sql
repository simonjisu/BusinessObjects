select count(*) as order_count
from orders o
where o.o_orderdate >= '1993-01-01'
  and o.o_orderdate < '1994-01-01'
  and exists (
    select *
    from lineitem l
    where l.l_orderkey = o.o_orderkey
      and l.l_commitdate < l.l_receiptdate
  );