# GLOBAL_SALES_OPPORTUNITY
# requires :1 list of n_names
select
	n_name,
	count(*) as numcust,
	sum(c_acctbal) as totacctbal
from
	(
		POSITIVE_ACCOUNT_BALANCE_COUNTRYNAME(:1)
	) as custsale (n_name, c_acctbal)
group by
	n_name
order by
	n_name;