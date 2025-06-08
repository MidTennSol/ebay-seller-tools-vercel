from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

Base = declarative_base()

class Order(Base):
    __tablename__ = 'orders'
    
    # Primary key
    id = Column(Integer, primary_key=True)
    
    # Core eBay Order Fields (from actual CSV structure)
    sales_record_number = Column(String(50))  # "Sales Record Number"
    order_number = Column(String(100))        # "Order Number" 
    transaction_id = Column(String(100))      # "Transaction ID"
    
    # Item Information
    item_number = Column(String(100))         # "Item Number" (eBay item ID)
    item_title = Column(String(500))          # "Item Title"
    custom_label = Column(String(200))        # "Custom Label"
    quantity = Column(Integer, default=1)     # "Quantity"
    
    # Financial Data - Core for Profit Calculation
    sold_for = Column(Float, default=0.0)               # "Sold For" (item price)
    shipping_and_handling = Column(Float, default=0.0)   # "Shipping And Handling"
    ebay_collected_tax = Column(Float, default=0.0)      # "eBay Collected Tax"
    seller_collected_tax = Column(Float, default=0.0)    # "Seller Collected Tax"
    total_price = Column(Float, default=0.0)             # "Total Price"
    
    # Manual Input Fields for Profit Calculation
    cogs = Column(Float, default=0.0)                    # Cost of Goods Sold (manual input)
    shipping_cost = Column(Float, default=0.0)           # Actual shipping cost (manual input)
    
    # Calculated Field
    total_profit = Column(Float, default=0.0)            # Calculated: (sold_for + shipping_and_handling) - (ebay_collected_tax + cogs + shipping_cost)
    
    # Buyer Information
    buyer_username = Column(String(100))      # "Buyer Username"
    buyer_name = Column(String(200))          # "Buyer Name"
    buyer_email = Column(String(200))         # "Buyer Email"
    
    # Shipping Information
    ship_to_name = Column(String(200))        # "Ship To Name"
    ship_to_city = Column(String(100))        # "Ship To City"
    ship_to_state = Column(String(50))        # "Ship To State"
    ship_to_zip = Column(String(20))          # "Ship To Zip"
    ship_to_country = Column(String(100))     # "Ship To Country"
    
    # Dates (stored as strings to match CSV format)
    sale_date = Column(String(50))            # "Sale Date"
    paid_on_date = Column(String(50))         # "Paid On Date"
    shipped_on_date = Column(String(50))      # "Shipped On Date"
    
    # Shipping Details
    shipping_service = Column(String(200))    # "Shipping Service"
    tracking_number = Column(String(200))     # "Tracking Number"
    
    # eBay Features
    promoted_listings = Column(String(10))    # "Sold Via Promoted Listings"
    ebay_plus = Column(String(10))           # "eBay Plus"
    global_shipping = Column(String(10))     # "Global Shipping Program"
    
    # Additional eBay Fields (for completeness)
    item_location = Column(String(200))       # "Item Location"
    item_zip_code = Column(String(20))        # "Item Zip Code"
    item_country = Column(String(100))        # "Item Country"
    payment_method = Column(String(100))      # "Payment Method"
    feedback_left = Column(String(50))        # "Feedback Left"
    feedback_received = Column(String(50))    # "Feedback Received"
    
    # eBay API Integration Fields
    ebay_order_id = Column(String(50), unique=True)      # eBay API Order ID
    legacy_order_id = Column(String(50))                 # Legacy Order ID
    marketplace_id = Column(String(20), default='EBAY_US') # Marketplace
    order_fulfillment_status = Column(String(50))        # Order status
    order_payment_status = Column(String(50))            # Payment status
    sync_status = Column(String(20), default='MANUAL')   # MANUAL, SYNCED, ERROR
    last_sync_at = Column(DateTime)                       # Last API sync timestamp
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def calculate_profit(self):
        """Calculate total profit based on the formula:
        Total Profit = (Sold For + Shipping Paid) - (eBay Tax + COGS + Shipping Cost)
        """
        revenue = (self.sold_for or 0) + (self.shipping_and_handling or 0)
        costs = (self.ebay_collected_tax or 0) + (self.seller_collected_tax or 0) + (self.cogs or 0) + (self.shipping_cost or 0)
        self.total_profit = revenue - costs
        return self.total_profit
    
    def to_dict(self):
        """Convert order to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'sales_record_number': self.sales_record_number,
            'order_number': self.order_number,
            'item_title': self.item_title,
            'quantity': self.quantity,
            'sold_for': self.sold_for,
            'shipping_and_handling': self.shipping_and_handling,
            'ebay_collected_tax': self.ebay_collected_tax,
            'total_price': self.total_price,
            'cogs': self.cogs,
            'shipping_cost': self.shipping_cost,
            'total_profit': self.total_profit,
            'sale_date': self.sale_date,
            'buyer_username': self.buyer_username,
            'shipping_service': self.shipping_service,
            'ebay_order_id': self.ebay_order_id,
            'sync_status': self.sync_status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class EBayToken(Base):
    """eBay API OAuth Token Storage"""
    __tablename__ = 'ebay_tokens'
    
    id = Column(Integer, primary_key=True)
    seller_id = Column(String(100), nullable=False)              # eBay seller ID (removed unique=True to allow multiple tokens)
    access_token = Column(Text, nullable=False)                   # OAuth access token
    refresh_token = Column(Text, nullable=True)                   # OAuth refresh token (can be null)
    token_expires_at = Column(DateTime, nullable=False)           # Token expiration
    refresh_expires_at = Column(DateTime, nullable=True)          # Refresh token expiration
    scope = Column(Text, nullable=True)                           # OAuth scopes (can be null)
    token_type = Column(String(20), default='user')               # Token type (user/app)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'seller_id': self.seller_id,
            'token_expires_at': self.token_expires_at.isoformat() if self.token_expires_at else None,
            'scope': self.scope,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class EBaySeller(Base):
    """eBay Seller Profile Information"""
    __tablename__ = 'ebay_sellers'
    
    seller_id = Column(String(100), primary_key=True)            # eBay seller ID
    seller_username = Column(String(100), nullable=False)        # eBay username
    seller_email = Column(String(255))                           # Seller email
    registration_date = Column(DateTime)                         # eBay registration date
    marketplace_id = Column(String(20), default='EBAY_US')       # Primary marketplace
    store_name = Column(String(255))                             # eBay store name
    seller_level = Column(String(50))                            # Seller performance level
    feedback_score = Column(Integer, default=0)                  # Feedback score
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        return {
            'seller_id': self.seller_id,
            'seller_username': self.seller_username,
            'seller_email': self.seller_email,
            'marketplace_id': self.marketplace_id,
            'store_name': self.store_name,
            'seller_level': self.seller_level,
            'feedback_score': self.feedback_score,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

# Database Configuration
class DatabaseConfig:
    def __init__(self, db_path='ebay_seller_tools.db'):
        self.db_path = db_path
        self.engine = create_engine(f'sqlite:///{db_path}', echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def create_tables(self):
        """Create all database tables"""
        Base.metadata.create_all(bind=self.engine)
        print(f"Database tables created successfully at: {self.db_path}")
    
    def get_session(self):
        """Get a database session"""
        return self.SessionLocal()
    
    def drop_tables(self):
        """Drop all database tables (for development/testing)"""
        Base.metadata.drop_all(bind=self.engine)
        print("All database tables dropped")

# Initialize database
db_config = DatabaseConfig()

def init_database():
    """Initialize the database with tables"""
    db_config.create_tables()
    
    # Handle database migrations for existing databases
    try:
        # Check if we need to add missing columns to EBayToken table
        import sqlite3
        conn = sqlite3.connect(db_config.db_path)
        cursor = conn.cursor()
        
        # Check if refresh_expires_at column exists
        cursor.execute("PRAGMA table_info(ebay_tokens)")
        columns = [row[1] for row in cursor.fetchall()]
        
        if 'refresh_expires_at' not in columns:
            print("Adding refresh_expires_at column to ebay_tokens table...")
            cursor.execute("ALTER TABLE ebay_tokens ADD COLUMN refresh_expires_at DATETIME")
            
        if 'token_type' not in columns:
            print("Adding token_type column to ebay_tokens table...")
            cursor.execute("ALTER TABLE ebay_tokens ADD COLUMN token_type VARCHAR(20) DEFAULT 'user'")
        
        # Make refresh_token nullable if it isn't already
        # SQLite doesn't support ALTER COLUMN, so we'll handle null values in the application
        
        conn.commit()
        conn.close()
        print("Database migration completed successfully")
        
    except Exception as e:
        print(f"Database migration warning (non-critical): {e}")
    
    return db_config

def get_db_session():
    """Get a database session for use in routes"""
    return db_config.get_session() 