SELECT COALESCE(order_count.order_count, 0) AS number_of_orders, COUNT(customer.C_CUSTKEY) AS customer_count
FROM CUSTOMER AS customer
LEFT JOIN (SELECT O_CUSTKEY, COUNT(O_ORDERKEY) AS order_count
            FROM ORDERS
            GROUP BY O_CUSTKEY) AS order_count
ON customer.C_CUSTKEY = order_count.O_CUSTKEY
GROUP BY number_of_orders
ORDER BY number_of_orders;