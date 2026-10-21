-- stg_posts: one clean row per post.
-- Fixes: duplicate rows from overlapping API pulls, inconsistent platform casing,
-- UTC timestamps converted to Pacific local date and hour for scheduling analysis.
DROP TABLE IF EXISTS stg_posts;

CREATE TABLE stg_posts AS
WITH deduped AS (
    SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY post_id ORDER BY _loaded_at) AS rn
    FROM raw_posts
),

cleaned AS (
    SELECT
        post_id,
        REPLACE(LOWER(TRIM(account_handle)), '@', '')            AS brand,
        REPLACE(LOWER(TRIM(platform)), ' ', '_')                  AS platform,
        LOWER(TRIM(media_type))                                   AS format,
        caption,
        CAST(duration_sec AS DOUBLE)                              AS duration_sec,
        LOWER(TRIM(creator_collab_tier))                          AS creator_tier,
        published_at                                              AS published_at_utc,
        SUBSTR(published_at, 1, 10)                               AS utc_date,
        CAST(SUBSTR(published_at, 12, 2) AS INTEGER)              AS utc_hour
    FROM deduped
    WHERE rn = 1
)

SELECT
    c.post_id,
    c.brand,
    c.platform,
    c.format,
    c.caption,
    c.duration_sec,
    CASE WHEN c.duration_sec IS NULL AND c.format <> 'carousel' THEN 1 ELSE 0 END AS is_duration_missing,
    c.creator_tier,
    c.published_at_utc,
    CASE WHEN c.utc_hour < 7 THEN d.prev_date_key ELSE d.date_key END            AS published_date_local,
    (c.utc_hour + 17) % 24                                                       AS published_hour_local
FROM cleaned AS c
JOIN dim_date AS d
    ON d.date_key = c.utc_date
