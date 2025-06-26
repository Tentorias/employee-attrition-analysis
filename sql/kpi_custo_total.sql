SELECT
    SUM(CASE WHEN Attrition = 'Yes' THEN MonthlyIncome * 0.5 ELSE 0 END) AS custo_total_turnover
FROM
    employees;