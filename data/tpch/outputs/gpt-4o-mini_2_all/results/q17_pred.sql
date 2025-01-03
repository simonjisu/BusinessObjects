select sum(l_extendedprice * (1 - l_discount)) as revenue
from lineitem l
join part p on l.l_partkey = p.p_partkey
where p.p_brand = 'Brand#23'