select
	n_name,
	count(*) as numcust,
	sum(c_acctbal) as totacctbal
from
	(
		select
			n_name,
			c_acctbal
		from
			customer,
			nation
		where
			c_nationkey = n_nationkey
			and n_name in ('SAUDI ARABIA', 'VIETNAM', 'RUSSIA', 'UNITED KINGDOM', 'UNITED STATES')
			and c_acctbal > (
				select
					avg(c_acctbal)
				from
					customer,
					nation
				where
					c_nationkey = n_nationkey
					and c_acctbal > 0.00
					and n_name in ('SAUDI ARABIA', 'VIETNAM', 'RUSSIA', 'UNITED KINGDOM', 'UNITED STATES')
			)
			and not exists (
				select
					*
				from
					orders
				where
					o_custkey = c_custkey
			)
	) as custsale
group by
	n_name
order by
	n_name;