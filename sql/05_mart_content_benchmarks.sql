-- mart_content_benchmarks: what kind of content works where.
-- Breakout rate is the headline metric because raw views are dominated by a few outliers.
DROP TABLE IF EXISTS mart_content_benchmarks;

CREATE TABLE mart_content_benchmarks AS
SELECT
    platform,
    theme,
    hook,
    COUNT(*)                                                    AS posts,
    AVG(views)                                                  AS avg_views,
    AVG(views_vs_baseline)                                      AS avg_views_vs_baseline,
    AVG(engagement_rate)                                        AS avg_engagement_rate,
    AVG(CAST(is_breakout AS DOUBLE))                            AS breakout_rate
FROM fct_post_performance
WHERE is_breakout IS NOT NULL
GROUP BY platform, theme, hook
