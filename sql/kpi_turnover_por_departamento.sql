-- KPI: Taxa de Turnover (%) e Contagem de Saídas por Departamento
-- Descrição: Identifica os departamentos com as maiores taxas de rotatividade.
--            Ajuda a focar os esforços de retenção onde eles são mais necessários.

SELECT
    Department,
    COUNT(*) AS total_funcionarios,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) AS total_saidas,
    ROUND((SUM(CASE WHEN Attrition = 'Yes' THEN 1.0 ELSE 0.0 END) * 100) / COUNT(*), 2) AS taxa_turnover_percent
FROM
    employees
GROUP BY
    Department
ORDER BY
    taxa_turnover_percent DESC;