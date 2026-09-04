/*
CREATE DATABASE AND TABLES FOR TIKI BOOK STORE
*/

-- Create database if not exists
IF NOT EXISTS (SELECT * FROM sys.databases WHERE name = 'TikiBookStore')
BEGIN
    CREATE DATABASE TikiBookStore;
    PRINT 'Database TikiBookStore created';
END
GO

USE TikiBookStore;
GO

-- Create book_recommendations table if not exists
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'book_recommendations')
BEGIN
    CREATE TABLE book_recommendations (
        source_id INT NOT NULL,
        source_name NVARCHAR(MAX),
        recommended_id INT NOT NULL,
        recommended_name NVARCHAR(MAX),
        similarity_score FLOAT,
        confidence_level NVARCHAR(10)
    );
    
    PRINT 'Table book_recommendations created';
END
ELSE
BEGIN
    PRINT 'Table book_recommendations already exists';
END

-- Add confidence_level column if not exists
IF NOT EXISTS (
    SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS 
    WHERE TABLE_NAME = 'book_recommendations' 
    AND COLUMN_NAME = 'confidence_level'
)
BEGIN
    ALTER TABLE book_recommendations
    ADD confidence_level NVARCHAR(10) NULL;
    
    PRINT 'Column confidence_level added';
END
ELSE
BEGIN
    PRINT 'Column confidence_level already exists';
END

-- Truncate old data
TRUNCATE TABLE book_recommendations;
PRINT 'Table truncated - ready for new data';

-- Show table structure
PRINT '';
PRINT 'Table structure:';
SELECT 
    COLUMN_NAME,
    DATA_TYPE,
    IS_NULLABLE
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME = 'book_recommendations'
ORDER BY ORDINAL_POSITION;

PRINT '';
PRINT '✓ Database and table ready for data load';
