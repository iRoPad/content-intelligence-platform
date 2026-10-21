-- mart_mmm_weekly: model-ready weekly table for marketing mix modeling.
-- Paid spend by channel + conversions, with organic views as a control variable.
DROP TABLE IF EXISTS mart_mmm_weekly;

CREATE TABLE mart_mmm_weekly AS
WITH organic AS (
    SELECT week_start, SUM(views) AS organic_views, COUNT(*) AS organic_posts
    FROM fct_post_performance
    GROUP BY week_start
)

SELECT
    pm.week_start,
    pm.tiktok_ads,
    pm.meta_ads,
    pm.youtube_ads,
    pm.creator_partnerships,
    pm.paid_search,
    pm.conversions,
    COALESCE(o.organic_views, 0) AS organic_views,
    COALESCE(o.organic_posts, 0) AS organic_posts
FROM raw_paid_media AS pm
LEFT JOIN organic AS o
    ON o.week_start = pm.week_start
ORDER BY pm.week_start
