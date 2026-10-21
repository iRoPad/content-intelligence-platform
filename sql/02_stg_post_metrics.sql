-- stg_post_metrics: lifetime (latest valid snapshot) and early-velocity metrics per post.
-- Drops API glitch rows (negative views) before picking the latest snapshot.
DROP TABLE IF EXISTS stg_post_metrics;

CREATE TABLE stg_post_metrics AS
WITH valid AS (
    SELECT *
    FROM raw_post_metrics
    WHERE views >= 0
),

latest AS (
    SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY post_id ORDER BY snapshot_day DESC) AS rn
    FROM valid
),

day1 AS (
    SELECT post_id, MAX(views) AS views_day1
    FROM valid
    WHERE snapshot_day = 1
    GROUP BY post_id
)

SELECT
    l.post_id,
    l.snapshot_day     AS days_tracked,
    l.views,
    l.likes,
    l.comments,
    l.shares,
    l.saves,
    l.watch_time_sec,
    d.views_day1
FROM latest AS l
LEFT JOIN day1 AS d
    ON d.post_id = l.post_id
WHERE l.rn = 1
