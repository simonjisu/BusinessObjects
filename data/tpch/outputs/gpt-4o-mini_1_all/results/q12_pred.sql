SELECT COUNT(*) AS late_lineitem_count FROM LINEITEM T1 WHERE strftime('%Y', T1.L_RECEIPTDATE) = '1993' AND T1.L_COMMITDATE < T1.L_RECEIPTDATE;