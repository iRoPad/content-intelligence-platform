-- fct_post_performance: the analytical grain. One row per post with content tags,
-- engagement ratios, and a leakage-free breakout label: views vs. the trailing
-- average of the previous 30 posts for the same brand x platform.
DROP TABLE IF EXISTS fct_post_performance;

CREATE TABLE fct_post_performance AS
WITH base AS (
    SELECT
        p.post_id,
        p.brand,
        p.platform,
        p.format,
        p.caption,
        p.duration_sec,
        p.is_duration_missing,
        p.creator_tier,
        p.published_at_utc,
        p.published_date_local,
        p.published_hour_local,
        d.week_start,
        d.month_key,
        d.weekday_num,
        d.is_weekend,
        t.theme,
        t.hook,
        t.tone,
        t.tag_source,
        m.days_tracked,
        m.views,
        m.views_day1,
        m.likes,
        m.comments,
        m.shares,
        m.saves,
        m.watch_time_sec,
        m.likes + m.comments + m.shares + m.saves AS engagements
    FROM stg_posts AS p
    JOIN stg_post_metrics AS m ON m.post_id = p.post_id
    JOIN dim_date AS d ON d.date_key = p.published_date_local
    LEFT JOIN raw_post_tags AS t ON t.post_id = p.post_id
),

windowed AS (
    SELECT
        *,
        AVG(views) OVER (
            PARTITION BY brand, platform ORDER BY published_at_utc
            ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING
        ) AS baseline_views,
        COUNT(*) OVER (
            PARTITION BY brand, platform ORDER BY published_at_utc
            ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING
        ) AS baseline_posts
    FROM base
)

SELECT
    *,
    CASE WHEN views > 0 THEN engagements * 1.0 / views END                         AS engagement_rate,
    CASE WHEN views > 0 THEN shares * 1.0 / views END                              AS share_rate,
    CASE WHEN views > 0 AND duration_sec > 0
         THEN watch_time_sec / (views * duration_sec) END                          AS avg_pct_viewed,
    CASE WHEN baseline_views > 0 THEN views * 1.0 / baseline_views END             AS views_vs_baseline,
    CASE
        WHEN baseline_posts < 10 THEN NULL
        WHEN views >= {{ breakout_multiple }} * baseline_views THEN 1
        ELSE 0
    END                                                                            AS is_breakout
FROM windowed
