-- mart_brand_weekly: weekly scorecard by brand and platform for the dashboard.
DROP TABLE IF EXISTS mart_brand_weekly;

CREATE TABLE mart_brand_weekly AS
SELECT
    week_start,
    brand,
    platform,
    COUNT(*)                                   AS posts,
    SUM(views)                                 AS views,
    SUM(engagements)                           AS engagements,
    SUM(engagements) * 1.0 / NULLIF(SUM(views), 0) AS engagement_rate,
    SUM(CASE WHEN is_breakout = 1 THEN 1 ELSE 0 END) AS breakouts,
    SUM(views) * 1.0 / COUNT(*)                AS views_per_post
FROM fct_post_performance
GROUP BY week_start, brand, platform
