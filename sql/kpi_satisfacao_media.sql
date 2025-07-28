    -- KPI: Níveis Médios de Satisfação por Nível de Cargo (JobLevel)
    -- Descrição: Analisa se há uma correlação entre o nível do cargo e a satisfação.
    --            Baixa satisfação em níveis mais altos pode indicar problemas de liderança ou carreira.

    SELECT
        JobLevel,
        CASE JobLevel
            WHEN 1 THEN 'Júnior'
            WHEN 2 THEN 'Pleno'
            WHEN 3 THEN 'Sênior'
            WHEN 4 THEN 'Gerente'
            WHEN 5 THEN 'Diretor'
            ELSE 'Outro'
        END AS nome_nivel_cargo,
        ROUND(AVG(JobSatisfaction), 2) AS media_satisfacao_trabalho,
        ROUND(AVG(EnvironmentSatisfaction), 2) AS media_satisfacao_ambiente
    FROM
        employees
    GROUP BY
        JobLevel
    ORDER BY
        JobLevel ASC;
