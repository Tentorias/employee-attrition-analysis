SELECT
    COUNT(*) AS total_funcionarios,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) AS total_saidas,
    ROUND(SUM(CASE WHEN Attrition = 'Yes' THEN 1.0 ELSE 0.0 END) / COUNT(*), 4) AS taxa_turnover_geral
FROM
    employees;
