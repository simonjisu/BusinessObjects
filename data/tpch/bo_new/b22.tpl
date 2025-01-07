# POSITIVE_ACCOUNT_BALANCE_COUNTRYNAME
# requires :1 list of n_names
select
	n_name,
	c_acctbal
from
	customer,
    nation
where
    c_nationkey = n_nationkey
	and n_name in (:1)
	and c_acctbal > (
		select
			avg(c_acctbal)
		from
			customer,
            nation
		where
			c_acctbal > 0.00
            and c_nationkey = n_nationkey
            and n_name in (:1)
	)
	and not exists (
		select
			*
		from
			orders
		where
			o_custkey = c_custkey
	)