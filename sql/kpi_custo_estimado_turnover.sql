-- KPI: Custo Estimado do Turnover por Departamento
-- Descrição: Estima o impacto financeiro da rotatividade em cada departamento.
-- Premissa: O custo de reposição de um funcionário é estimado em 50% do seu salário mensal.
--            (Esta é uma premissa simplificada que pode ser ajustada).

SELECT
    Department,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) AS total_saidas,
    ROUND(AVG(MonthlyIncome), 2) AS media_salario_mensal,
    ROUND(SUM(CASE WHEN Attrition = 'Yes' THEN MonthlyIncome * 0.5 ELSE 0 END), 2) AS custo_estimado_turnover
FROM
    employees
GROUP BY
    Department
ORDER BY
    custo_estimado_turnover DESC;