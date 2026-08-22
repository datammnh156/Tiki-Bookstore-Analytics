USE TikiBookStore;
GO

-- Xóa bảng cũ nếu tồn tại (theo thứ tự Foreign Key)
IF OBJECT_ID('shap_values', 'U') IS NOT NULL DROP TABLE shap_values;
IF OBJECT_ID('book_recommendations', 'U') IS NOT NULL DROP TABLE book_recommendations;
IF OBJECT_ID('discount_recommendations', 'U') IS NOT NULL DROP TABLE discount_recommendations;
IF OBJECT_ID('books', 'U') IS NOT NULL DROP TABLE books;
IF OBJECT_ID('dim_category', 'U') IS NOT NULL DROP TABLE dim_category;
GO

-- ============================================================================
-- 1. BẢNG DIM_CATEGORY - Danh mục sách
-- ============================================================================
CREATE TABLE dim_category (
    category_id INT PRIMARY KEY,
    category_name NVARCHAR(100)
);
GO

-- ============================================================================
-- 2. BẢNG BOOKS - Thông tin sách
-- ============================================================================
CREATE TABLE books (
    id INT PRIMARY KEY,
    category_id INT FOREIGN KEY REFERENCES dim_category(category_id),
    name NVARCHAR(500),
    price FLOAT,
    original_price FLOAT,
    discount_rate FLOAT,
    rating_average FLOAT,
    review_count INT,
    quantity_sold INT,
    favourite_count INT,
    has_rating INT,
    is_bestseller INT
);
GO

-- ============================================================================
-- 3. BẢNG DISCOUNT_RECOMMENDATIONS - Gợi ý giảm giá tối ưu
-- ============================================================================
CREATE TABLE discount_recommendations (
    id INT PRIMARY KEY,
    current_discount FLOAT,
    current_probability FLOAT,
    optimal_discount FLOAT,
    optimal_probability FLOAT,
    improvement FLOAT,
    recommend_change BIT,
    FOREIGN KEY (id) REFERENCES books(id)
);
GO

-- ============================================================================
-- 4. BẢNG BOOK_RECOMMENDATIONS - Gợi ý sách liên quan
-- ============================================================================
CREATE TABLE book_recommendations (
    source_id INT,
    recommended_id INT,
    similarity_score FLOAT,
    PRIMARY KEY (source_id, recommended_id),
    FOREIGN KEY (source_id) REFERENCES books(id),
    FOREIGN KEY (recommended_id) REFERENCES books(id)
);
GO

-- ============================================================================
-- 5. BẢNG SHAP_VALUES - Feature importance từ ML model
-- ============================================================================
CREATE TABLE shap_values (
    shap_id INT IDENTITY PRIMARY KEY,
    book_id INT FOREIGN KEY REFERENCES books(id),
    feature_name NVARCHAR(100),
    shap_value FLOAT,
    feature_value NVARCHAR(100),
    shap_value_percent FLOAT
);
GO

-- ============================================================================
-- TẠO INDEX ĐỂ TỐI ƯU QUERY
-- ============================================================================
CREATE INDEX idx_books_category ON books(category_id);
CREATE INDEX idx_books_bestseller ON books(is_bestseller);
CREATE INDEX idx_discount_recommend ON discount_recommendations(recommend_change);
CREATE INDEX idx_shap_book ON shap_values(book_id);
GO

PRINT '✓ Đã tạo xong 5 bảng và các index trong database TikiBookStore';
GO