/*
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  UPDATE book_recommendations TABLE FOR HYBRID SYSTEM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Purpose: Update table structure and clear old data for new hybrid recommendations
  
  Changes:
    1. Add confidence_level column (NVARCHAR(10))
    2. Truncate old data
    3. Ready for new data load
  
  Date: 2026-08-25
  Author: Kiro
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
*/

USE TikiBookStore;
GO

PRINT '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━';
PRINT 'STEP 1: Check current table structure';
PRINT '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━';

-- Check if confidence_level column exists
IF EXISTS (
    SELECT 1 
    FROM INFORMATION_SCHEMA.COLUMNS 
    WHERE TABLE_NAME = 'book_recommendations' 
    AND COLUMN_NAME = 'confidence_level'
)
BEGIN
    PRINT '✓ Column confidence_level already exists';
END
ELSE
BEGIN
    PRINT '○ Column confidence_level does not exist - will be added';
END

-- Check current row count
DECLARE @OldCount INT;
SELECT @OldCount = COUNT(*) FROM book_recommendations;
PRINT CONCAT('  Current rows in book_recommendations: ', @OldCount);

PRINT '';
PRINT '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━';
PRINT 'STEP 2: Add confidence_level column (if not exists)';
PRINT '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━';

IF NOT EXISTS (
    SELECT 1 
    FROM INFORMATION_SCHEMA.COLUMNS 
    WHERE TABLE_NAME = 'book_recommendations' 
    AND COLUMN_NAME = 'confidence_level'
)
BEGIN
    ALTER TABLE book_recommendations
    ADD confidence_level NVARCHAR(10) NULL;
    
    PRINT '✓ Added column: confidence_level NVARCHAR(10)';
END
ELSE
BEGIN
    PRINT '✓ Column confidence_level already exists - skipped';
END

PRINT '';
PRINT '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━';
PRINT 'STEP 3: Clear old data (TRUNCATE)';
PRINT '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━';

-- Truncate old recommendations
TRUNCATE TABLE book_recommendations;
PRINT '✓ Truncated book_recommendations table';

-- Verify empty
SELECT @OldCount = COUNT(*) FROM book_recommendations;
PRINT CONCAT('  Current rows after truncate: ', @OldCount);

IF @OldCount = 0
BEGIN
    PRINT '✓ Table is empty and ready for new data';
END
ELSE
BEGIN
    PRINT '⚠ Warning: Table still has rows!';
END

PRINT '';
PRINT '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━';
PRINT 'STEP 4: Show table structure';
PRINT '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━';

SELECT 
    COLUMN_NAME,
    DATA_TYPE,
    CHARACTER_MAXIMUM_LENGTH,
    IS_NULLABLE
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME = 'book_recommendations'
ORDER BY ORDINAL_POSITION;

PRINT '';
PRINT '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━';
PRINT '✅ UPDATE COMPLETE - Ready for new data load';
PRINT '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━';
PRINT '';
PRINT 'Next steps:';
PRINT '  1. Run: python scripts/etl/load_to_sql.py';
PRINT '  2. Verify: SELECT COUNT(*) FROM book_recommendations';
PRINT '  3. Check: SELECT confidence_level, COUNT(*) GROUP BY confidence_level';
PRINT '';

GO
