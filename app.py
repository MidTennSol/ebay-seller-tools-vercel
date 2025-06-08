from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from models import get_db_session, Order, init_database, EBayToken, EBaySeller
import io
import csv
import logging
import re
from decimal import Decimal, InvalidOperation
from urllib.parse import parse_qs, urlparse
import requests
from sqlalchemy import func, desc
import sqlite3

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize database on startup
logger.info("Initializing database...")
init_database()
logger.info("Database initialized successfully")

# Import the eBay inventory service
try:
    from ebay_inventory_service import EbayInventoryService
    ebay_inventory_service = EbayInventoryService(environment="sandbox")
    print("✅ eBay Inventory Service imported successfully")
    logger.info(f"eBay service available: {ebay_inventory_service.is_available()}")
except Exception as e:
    print(f"⚠️  Warning: Could not import eBay Inventory Service: {e}")
    logger.error(f"eBay Inventory Service import error: {e}")
    ebay_inventory_service = None

# Initialize eBay OAuth service
try:
    from ebay_oauth_service import EbayOAuthService
    oauth_service = EbayOAuthService(environment="sandbox")
    print("✅ eBay OAuth Service imported successfully")
    logger.info("eBay OAuth service available")
except Exception as e:
    print(f"⚠️  Warning: Could not import eBay OAuth Service: {e}")
    logger.error(f"eBay OAuth Service import error: {e}")
    oauth_service = None

def validate_numeric_input(value, field_name, min_value=None, max_value=None):
    """Validate numeric input with detailed error messages"""
    if value is None or value == '':
        return 0.0
    
    try:
        # Handle string inputs that might contain currency symbols
        if isinstance(value, str):
            # Handle pandas NaN strings and other empty values
            if str(value).lower() in ['nan', 'none', 'null', '--', 'n/a']:
                return 0.0
                
            # Remove common currency symbols and whitespace
            cleaned_value = re.sub(r'[$,\s]', '', str(value))
            if cleaned_value == '' or cleaned_value == '-':
                return 0.0
            numeric_value = float(cleaned_value)
        else:
            numeric_value = float(value)
        
        # Check for NaN result
        if pd.isna(numeric_value):
            return 0.0
            
        # Validate range if specified
        if min_value is not None and numeric_value < min_value:
            raise ValueError(f"{field_name} cannot be less than {min_value}")
        
        if max_value is not None and numeric_value > max_value:
            raise ValueError(f"{field_name} cannot be greater than {max_value}")
        
        # Round to 2 decimal places for currency
        return round(numeric_value, 2)
        
    except (ValueError, TypeError, InvalidOperation) as e:
        # If we can't parse it, default to 0.0 for financial fields
        logger.warning(f"Could not parse {field_name} value '{value}', defaulting to 0.0: {str(e)}")
        return 0.0

def validate_csv_structure(df):
    """Validate CSV structure and provide detailed feedback"""
    errors = []
    warnings = []
    
    # Check for empty CSV
    if len(df) == 0:
        errors.append({
            'type': 'empty_file',
            'message': "CSV file is empty or contains no data rows",
            'details': "Please ensure your CSV file contains eBay order data"
        })
        return errors, warnings
    
    # Log all columns for debugging
    logger.info(f"All CSV columns: {list(df.columns)}")
    
    # Check if we have all unnamed columns (header parsing issue)
    unnamed_cols = [col for col in df.columns if str(col).startswith('Unnamed:')]
    if len(unnamed_cols) > len(df.columns) * 0.8:  # If 80%+ columns are unnamed
        errors.append({
            'type': 'header_parsing_issue',
            'message': "CSV headers could not be parsed correctly",
            'details': f"Found {len(unnamed_cols)} unnamed columns out of {len(df.columns)} total. This usually means the CSV file doesn't have proper headers in the first row, or the file format is not standard. Please check that your eBay CSV export has column headers like 'Sales Record Number', 'Order Number', etc."
        })
        return errors, warnings
    
    # Required columns for eBay CSV processing (flexible matching)
    required_columns_patterns = {
        'sales_record': ['Sales Record Number', 'sales record number', 'Record Number', 'Transaction ID'],
        'order_number': ['Order Number', 'order number', 'Order ID'],
        'item_title': ['Item Title', 'item title', 'Title', 'Product', 'Item'],
        'sold_for': ['Sold For', 'sold for', 'Sale Price', 'Price', 'Item Price'],
        'total_price': ['Total Price', 'total price', 'Total', 'Final Price']
    }
    
    # Additional eBay columns we can use for processing
    optional_columns_patterns = {
        'shipping_and_handling': ['Shipping and Handling', 'Shipping & Handling', 'Shipping Cost', 'Shipping'],
        'ebay_collected_tax': ['eBay Collected Tax', 'eBay Tax', 'Tax', 'eBay Collected Tax and Fees Included in Total'],
        'sale_date': ['Sale Date', 'Date Sold', 'Order Date'],
        'payment_method': ['Payment Method'],
        'tracking_number': ['Tracking Number', 'Tracking'],
        'shipping_service': ['Shipping Service'],
        'buyer_username': ['Buyer Username', 'Buyer', 'Username'],
        'quantity': ['Quantity', 'Qty']
    }
    
    found_columns = {}
    missing_patterns = []
    
    # Try to find matching columns using flexible patterns
    for key, patterns in required_columns_patterns.items():
        found = False
        for pattern in patterns:
            matching_cols = [col for col in df.columns if pattern.lower() in str(col).lower()]
            if matching_cols:
                found_columns[key] = matching_cols[0]
                found = True
                break
        
        if not found:
            missing_patterns.append(patterns[0])  # Use the primary pattern name
    
    # Also check for optional columns
    for key, patterns in optional_columns_patterns.items():
        for pattern in patterns:
            matching_cols = [col for col in df.columns if pattern.lower() in str(col).lower()]
            if matching_cols:
                found_columns[key] = matching_cols[0]
                break
    
    # For eBay CSVs, we need at least some key columns, but can be more flexible
    essential_columns = ['total_price', 'sale_date']  # These are usually always present
    missing_essential = [col for col in essential_columns if col not in found_columns]
    
    # If we're missing ALL essential columns, it's probably not a valid eBay CSV
    if len(missing_essential) == len(essential_columns):
        errors.append({
            'type': 'missing_columns',
            'message': f"Could not identify essential eBay columns. Missing: {', '.join(missing_patterns)}",
            'details': f"Available columns: {', '.join(list(df.columns)[:10])}{'...' if len(df.columns) > 10 else ''}. Please ensure this is a valid eBay orders export CSV file."
        })
    elif missing_patterns:
        # Just warnings for missing non-essential columns
        warnings.append({
            'type': 'missing_optional_columns',
            'message': f"Some optional columns not found: {', '.join(missing_patterns)}",
            'details': f"Processing will continue with available columns: {', '.join(found_columns.keys())}"
        })
    
    # Check for duplicate sales record numbers if we found the column
    if 'sales_record' in found_columns:
        sales_col = found_columns['sales_record']
        sales_records = df[sales_col].dropna()
        duplicates = sales_records[sales_records.duplicated()].unique()
        if len(duplicates) > 0:
            warnings.append({
                'type': 'duplicate_records',
                'message': f"Found {len(duplicates)} duplicate Sales Record Numbers",
                'details': f"Duplicates will be skipped: {', '.join(map(str, duplicates[:5]))}"
            })
    
    # Check for missing critical data
    if 'sold_for' in found_columns:
        sold_col = found_columns['sold_for']
        missing_sold_for = df[sold_col].isna().sum()
        if missing_sold_for > 0:
            warnings.append({
                'type': 'missing_data',
                'message': f"{missing_sold_for} rows missing 'Sold For' data",
                'details': "These rows will have $0.00 for sold price"
            })
    
    return errors, warnings

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint with detailed system status"""
    session = get_db_session()
    try:
        order_count = session.query(Order).count()
        
        # Calculate total profit correctly
        total_profit = sum(order.total_profit or 0 for order in session.query(Order).all())
        
        logger.info(f"Health check: {order_count} orders, ${total_profit:.2f} total profit")
        
        return jsonify({
            'status': 'healthy', 
            'message': 'eBay Seller Tools API is running',
            'database': 'connected',
            'orders_count': order_count,
            'total_profit': round(float(total_profit), 2),
            'version': '1.0.0',
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Database connection failed',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500
    finally:
        session.close()

@app.route('/api/orders', methods=['GET'])
def get_orders():
    """Get all orders with optional filtering and enhanced validation"""
    session = get_db_session()
    
    try:
        # Validate query parameters
        missing_cogs = request.args.get('missing_cogs', 'false').lower()
        negative_profit = request.args.get('negative_profit', 'false').lower()
        
        if missing_cogs not in ['true', 'false']:
            return jsonify({'error': 'missing_cogs parameter must be true or false'}), 400
        
        if negative_profit not in ['true', 'false']:
            return jsonify({'error': 'negative_profit parameter must be true or false'}), 400
        
        missing_cogs = missing_cogs == 'true'
        negative_profit = negative_profit == 'true'
        
        # Base query
        query = session.query(Order)
        
        # Apply filters
        if missing_cogs:
            query = query.filter(Order.cogs == 0.0)
            logger.info("Applied missing COGS filter")
        
        if negative_profit:
            query = query.filter(Order.total_profit < 0)
            logger.info("Applied negative profit filter")
        
        # Get orders and convert to list of dicts
        orders = query.order_by(Order.created_at.desc()).all()
        orders_data = [order.to_dict() for order in orders]
        
        # Calculate comprehensive summary statistics
        total_orders = len(orders_data)
        total_profit = sum(order['total_profit'] or 0 for order in orders_data)
        negative_orders = len([o for o in orders_data if (o['total_profit'] or 0) < 0])
        missing_cogs_count = len([o for o in orders_data if (o['cogs'] or 0) == 0])
        profitable_orders = len([o for o in orders_data if (o['total_profit'] or 0) > 0])
        
        # Calculate average metrics if we have orders
        avg_profit = total_profit / total_orders if total_orders > 0 else 0
        avg_selling_price = sum(order['sold_for'] or 0 for order in orders_data) / total_orders if total_orders > 0 else 0
        
        logger.info(f"Retrieved {total_orders} orders, total profit: ${total_profit:.2f}")
        
        return jsonify({
            'orders': orders_data,
            'summary': {
                'total_orders': total_orders,
                'total_profit': round(total_profit, 2),
                'average_profit': round(avg_profit, 2),
                'average_selling_price': round(avg_selling_price, 2),
                'profitable_orders': profitable_orders,
                'negative_profit_orders': negative_orders,
                'missing_cogs_orders': missing_cogs_count,
                'completion_rate': round((total_orders - missing_cogs_count) / total_orders * 100, 1) if total_orders > 0 else 0
            },
            'filters_applied': {
                'missing_cogs': missing_cogs,
                'negative_profit': negative_profit
            }
        })
        
    except Exception as e:
        logger.error(f"Failed to fetch orders: {str(e)}")
        return jsonify({'error': f'Failed to fetch orders: {str(e)}'}), 500
    finally:
        session.close()

@app.route('/api/orders/<int:order_id>', methods=['PUT'])
def update_order(order_id):
    """Update order with comprehensive input validation"""
    session = get_db_session()
    
    try:
        # Validate order_id
        if order_id <= 0:
            return jsonify({'error': 'Order ID must be a positive integer'}), 400
        
        # Get the order
        order = session.query(Order).filter(Order.id == order_id).first()
        if not order:
            logger.warning(f"Order not found: {order_id}")
            return jsonify({'error': f'Order with ID {order_id} not found'}), 404
        
        # Get update data
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided. Please include COGS and/or shipping_cost in request body.'}), 400
        
        # Store original values for logging
        original_cogs = order.cogs
        original_shipping = order.shipping_cost
        original_profit = order.total_profit
        
        # Validate and update allowed fields
        if 'cogs' in data:
            try:
                order.cogs = validate_numeric_input(data['cogs'], 'COGS', min_value=0, max_value=10000)
            except ValueError as e:
                return jsonify({'error': str(e)}), 400
        
        if 'shipping_cost' in data:
            try:
                order.shipping_cost = validate_numeric_input(data['shipping_cost'], 'Shipping Cost', min_value=0, max_value=1000)
            except ValueError as e:
                return jsonify({'error': str(e)}), 400
        
        # Recalculate profit
        order.calculate_profit()
        order.updated_at = datetime.utcnow()
        
        # Commit changes
        session.commit()
        
        # Log the update
        logger.info(f"Updated order {order_id}: COGS ${original_cogs:.2f} -> ${order.cogs:.2f}, "
                   f"Shipping ${original_shipping:.2f} -> ${order.shipping_cost:.2f}, "
                   f"Profit ${original_profit:.2f} -> ${order.total_profit:.2f}")
        
        return jsonify({
            'message': 'Order updated successfully',
            'order': order.to_dict(),
            'changes': {
                'cogs_changed': original_cogs != order.cogs,
                'shipping_changed': original_shipping != order.shipping_cost,
                'profit_change': round(order.total_profit - original_profit, 2)
            }
        })
        
    except ValueError as e:
        logger.error(f"Validation error updating order {order_id}: {str(e)}")
        return jsonify({'error': f'Validation error: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"Failed to update order {order_id}: {str(e)}")
        session.rollback()
        return jsonify({'error': f'Failed to update order: {str(e)}'}), 500
    finally:
        session.close()

@app.route('/api/debug-csv', methods=['POST'])
def debug_csv():
    """Debug endpoint to analyze CSV structure without processing"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if not file.filename.lower().endswith('.csv'):
        return jsonify({'error': 'Not a CSV file'}), 400
    
    try:
        # Try multiple ways to read the CSV
        debug_info = {}
        
        # Method 1: Default pandas read
        file.seek(0)
        try:
            df1 = pd.read_csv(file)
            debug_info['default_read'] = {
                'columns': list(df1.columns)[:10],
                'shape': df1.shape,
                'first_row': df1.iloc[0].to_dict() if len(df1) > 0 else {}
            }
        except Exception as e:
            debug_info['default_read'] = {'error': str(e)}
        
        # Method 2: Read raw content
        file.seek(0)
        raw_content = file.read().decode('utf-8', errors='ignore')
        lines = raw_content.split('\n')[:10]
        debug_info['raw_content'] = {
            'first_10_lines': lines,
            'total_lines': len(raw_content.split('\n')),
            'encoding_detected': 'utf-8'
        }
        
        # Method 3: Try different header rows
        file.seek(0)
        debug_info['header_attempts'] = {}
        for header_row in [0, 1, 2, 3]:
            try:
                file.seek(0)
                df_header = pd.read_csv(file, header=header_row)
                debug_info['header_attempts'][f'header_{header_row}'] = {
                    'columns': list(df_header.columns)[:10],
                    'shape': df_header.shape
                }
            except Exception as e:
                debug_info['header_attempts'][f'header_{header_row}'] = {'error': str(e)}
        
        return jsonify(debug_info)
        
    except Exception as e:
        return jsonify({'error': f'Debug failed: {str(e)}'}), 500

@app.route('/api/upload', methods=['POST'])
def upload_csv():
    """Upload and parse eBay CSV file with comprehensive validation"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided. Please select a CSV file to upload.'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected. Please choose a file.'}), 400
    
    if not file.filename.lower().endswith('.csv'):
        return jsonify({'error': f'Invalid file type: {file.filename}. Only CSV files are supported.'}), 400
    
    # Validate file size
    file.seek(0, 2)  # Seek to end
    file_size = file.tell()
    file.seek(0)  # Reset to beginning
    
    if file_size == 0:
        return jsonify({'error': 'File is empty. Please upload a file with data.'}), 400
    
    if file_size > app.config['MAX_CONTENT_LENGTH']:
        return jsonify({'error': f'File too large ({file_size} bytes). Maximum size is {app.config["MAX_CONTENT_LENGTH"]} bytes.'}), 400
    
    session = get_db_session()
    
    try:
        logger.info(f"Processing CSV upload: {file.filename} ({file_size} bytes)")
        
        # Read CSV file with error handling
        try:
            # Read raw file content first to analyze structure
            file.seek(0)
            raw_content = file.read().decode('utf-8')
            lines = raw_content.split('\n')
            
            logger.info(f"Analyzing CSV structure - first 5 lines:")
            for i, line in enumerate(lines[:5]):
                logger.info(f"Line {i}: {line[:100]}...")
            
            # Reset file pointer
            file.seek(0)
            
            # Special handling for eBay CSV format
            # eBay CSVs often have: empty row, headers, empty row, data...
            header_row = None
            skip_rows = None
            
            # Look for the header row (line containing eBay column names)
            ebay_headers = ['Sales Record Number', 'Order Number', 'Item Title', 'Total Price', 'Sale Date']
            
            for i, line in enumerate(lines[:5]):
                if any(header in line for header in ebay_headers):
                    logger.info(f"Found eBay headers on line {i}: {line[:200]}...")
                    header_row = i
                    break
            
            if header_row is not None:
                logger.info(f"Using line {header_row} as header row")
                # Read CSV with the detected header row
                file.seek(0)
                csv_data = pd.read_csv(file, header=header_row)
                
                logger.info(f"Raw CSV shape after reading: {csv_data.shape}")
                logger.info(f"First few rows before cleaning:")
                for i in range(min(3, len(csv_data))):
                    sales_record = csv_data.iloc[i].get('Sales Record Number', 'N/A')
                    item_title = csv_data.iloc[i].get('Item Title', 'N/A')
                    logger.info(f"  Row {i}: Sales Record='{sales_record}', Item Title='{item_title}'")
                
                # Remove any empty rows after the header
                csv_data = csv_data.dropna(how='all')  # Remove completely empty rows
                
                # Remove rows where Sales Record Number is empty, NaN, or just empty strings
                if 'Sales Record Number' in csv_data.columns:
                    # Log before filtering
                    logger.info(f"Before Sales Record filtering: {len(csv_data)} rows")
                    
                    # Remove NaN values
                    csv_data = csv_data[csv_data['Sales Record Number'].notna()]
                    
                    # Remove empty strings and whitespace-only strings
                    csv_data = csv_data[csv_data['Sales Record Number'].astype(str).str.strip() != '']
                    csv_data = csv_data[csv_data['Sales Record Number'].astype(str).str.strip() != 'nan']
                    
                    # Log after filtering
                    logger.info(f"After Sales Record filtering: {len(csv_data)} rows")
                
                logger.info(f"After cleaning, CSV shape: {csv_data.shape}")
                logger.info(f"CSV columns: {list(csv_data.columns)}")
                
                # Log first few rows after cleaning
                if len(csv_data) > 0:
                    logger.info(f"First row after cleaning:")
                    first_row = csv_data.iloc[0]
                    sales_record = first_row.get('Sales Record Number', 'N/A')
                    item_title = first_row.get('Item Title', 'N/A')
                    logger.info(f"  Sales Record: '{sales_record}', Item Title: '{item_title}'")
                
            else:
                # Fall back to standard CSV reading
                logger.info("No eBay headers detected, using standard CSV parsing")
                file.seek(0)
                csv_data = pd.read_csv(file)
                
                # Check if we have unnamed columns (indicating header issues)
                unnamed_cols = [col for col in csv_data.columns if str(col).startswith('Unnamed:')]
                if len(unnamed_cols) > 5:
                    logger.warning("Detected many unnamed columns, trying header detection...")
                    
                    # Try different header rows
                    for header_attempt in [1, 2]:
                        try:
                            file.seek(0)
                            test_csv = pd.read_csv(file, header=header_attempt)
                            if 'Sales Record Number' in test_csv.columns or 'Order Number' in test_csv.columns:
                                csv_data = test_csv
                                logger.info(f"Successfully used header row {header_attempt}")
                                break
                        except Exception:
                            continue
            
        except pd.errors.EmptyDataError:
            return jsonify({'error': 'CSV file is empty or corrupted.'}), 400
        except pd.errors.ParserError as e:
            return jsonify({'error': f'CSV parsing error: {str(e)}. Please check file format.'}), 400
        except Exception as e:
            return jsonify({'error': f'Failed to read CSV file: {str(e)}'}), 400
        
        # Validate CSV structure
        validation_errors, validation_warnings = validate_csv_structure(csv_data)
        
        if validation_errors:
            logger.warning(f"CSV validation failed: {validation_errors}")
            return jsonify({
                'error': 'CSV validation failed',
                'validation_errors': validation_errors,
                'validation_warnings': validation_warnings
            }), 400

        # Create column mapping for flexible CSV processing
        def find_column(patterns):
            for pattern in patterns:
                for col in csv_data.columns:
                    if pattern.lower() in str(col).lower():
                        return col
            return None

        column_map = {
            'sales_record': find_column(['Sales Record Number', 'Transaction ID', 'Record Number']),
            'order_number': find_column(['Order Number', 'Order ID']),
            'item_title': find_column(['Item Title', 'Title', 'Product', 'Item']),
            'sold_for': find_column(['Sold For', 'Sale Price', 'Price', 'Item Price']),
            'total_price': find_column(['Total Price', 'Final Price', 'Total']),
            'shipping_and_handling': find_column(['Shipping and Handling', 'Shipping & Handling', 'Shipping Cost']),
            'ebay_collected_tax': find_column(['eBay Collected Tax', 'eBay Tax', 'Tax', 'eBay Collected Tax and Fees Included in Total']),
            'sale_date': find_column(['Sale Date', 'Date Sold', 'Order Date']),
            'payment_method': find_column(['Payment Method']),
            'tracking_number': find_column(['Tracking Number', 'Tracking']),
            'shipping_service': find_column(['Shipping Service']),
            'buyer_username': find_column(['Buyer Username', 'Buyer', 'Username']),
            'quantity': find_column(['Quantity', 'Qty'])
        }

        # Remove None values
        column_map = {k: v for k, v in column_map.items() if v is not None}
        logger.info(f"Column mapping: {column_map}")

        orders_created = 0
        orders_updated = 0
        orders_skipped = 0
        errors = []
        
        logger.info(f"Processing {len(csv_data)} rows from CSV")
        
        # Debug: Show first few rows of actual data
        if len(csv_data) > 0:
            logger.info(f"First row of data: {csv_data.iloc[0].to_dict()}")
            if len(csv_data) > 1:
                logger.info(f"Second row of data: {csv_data.iloc[1].to_dict()}")
        
        for index, row in csv_data.iterrows():
            try:
                # Skip empty rows or header rows
                sales_record = ''
                if 'sales_record' in column_map:
                    sales_record = row.get(column_map['sales_record'], '')
                elif 'order_number' in column_map:
                    sales_record = row.get(column_map['order_number'], '')
                    
                # Skip if sales record is empty or NaN
                if pd.isna(sales_record) or str(sales_record).strip() == '':
                    orders_skipped += 1
                    continue
                
                # Skip metadata rows (contains 'Seller ID', 'middletennsolutions', etc.)
                sales_record_str = str(sales_record).strip()
                if ('seller id' in sales_record_str.lower() or 
                    'middletennsolutions' in sales_record_str.lower() or
                    sales_record_str.startswith('---') or
                    sales_record_str.startswith('Total') or
                    'summary' in sales_record_str.lower()):
                    logger.info(f"Skipping metadata row {index}: '{sales_record_str}'")
                    orders_skipped += 1
                    continue
                
                # Get item title for validation
                item_title = ''
                if 'item_title' in column_map:
                    item_title = str(row.get(column_map['item_title'], '')).strip()
                
                # Skip rows with invalid item titles
                if (pd.isna(item_title) or 
                    item_title.lower() in ['nan', 'none', '', 'null'] or
                    len(item_title) < 3):
                    logger.info(f"Skipping row {index} with invalid item title: '{item_title}'")
                    orders_skipped += 1
                    continue
                
                # Validate that sales record is numeric (eBay sales records are numbers)
                try:
                    sales_record_num = int(float(str(sales_record).replace(',', '')))
                    if sales_record_num <= 0:
                        logger.info(f"Skipping row {index} with invalid sales record number: '{sales_record}'")
                        orders_skipped += 1
                        continue
                except (ValueError, TypeError):
                    logger.info(f"Skipping row {index} with non-numeric sales record: '{sales_record}'")
                    orders_skipped += 1
                    continue
                
                # Check if order already exists
                existing_order = session.query(Order).filter(
                    Order.sales_record_number == str(sales_record)
                ).first()
                
                if existing_order:
                    orders_updated += 1
                    continue
                
                # Get values using column mapping with defaults
                def get_value(key, default=''):
                    try:
                        if key in column_map and column_map[key] is not None:
                            value = row.get(column_map[key], default)
                            logger.debug(f"Row {index}: {key} = '{value}' (from column '{column_map[key]}')")
                            return value if value is not None else default
                        else:
                            logger.debug(f"Row {index}: {key} not found in column_map, using default '{default}'")
                        return default
                    except Exception as e:
                        logger.warning(f"Error getting value for {key}: {str(e)}")
                        return default
                
                # Validate and clean financial data
                try:
                    sold_for = validate_numeric_input(get_value('sold_for', '0'), 'Sold For', min_value=0)
                except Exception as e:
                    logger.warning(f"Row {index}: Error validating sold_for: {str(e)}")
                    sold_for = 0.0
                
                try:
                    shipping_handling = validate_numeric_input(get_value('shipping_and_handling', '0'), 'Shipping And Handling', min_value=0)
                except Exception as e:
                    logger.warning(f"Row {index}: Error validating shipping_handling: {str(e)}")
                    shipping_handling = 0.0
                
                try:
                    ebay_tax = validate_numeric_input(get_value('ebay_collected_tax', '0'), 'eBay Collected Tax', min_value=0)
                except Exception as e:
                    logger.warning(f"Row {index}: Error validating ebay_tax: {str(e)}")
                    ebay_tax = 0.0
                
                try:
                    total_price = validate_numeric_input(get_value('total_price', '0'), 'Total Price', min_value=0)
                except Exception as e:
                    logger.warning(f"Row {index}: Error validating total_price: {str(e)}")
                    total_price = 0.0
                
                # If we don't have sold_for but have total_price, use total_price as sold_for
                if sold_for == 0 and total_price > 0:
                    sold_for = total_price
                
                # Create new order with validated data
                order = Order(
                    sales_record_number=str(sales_record),
                    order_number=str(get_value('order_number')),
                    transaction_id=str(sales_record),  # Use sales_record as transaction_id if we don't have separate
                    item_number='',  # eBay item ID not in this CSV format
                    item_title=str(get_value('item_title'))[:500],  # Limit length
                    custom_label='',
                    quantity=max(1, int(get_value('quantity', 1))) if pd.notna(get_value('quantity')) else 1,
                    
                    # Financial data (validated)
                    sold_for=sold_for,
                    shipping_and_handling=shipping_handling,
                    ebay_collected_tax=ebay_tax,
                    seller_collected_tax=0.0,  # Not in this CSV format
                    total_price=total_price,
                    
                    # Buyer info
                    buyer_username=str(get_value('buyer_username'))[:100],
                    buyer_name='',  # Not in this CSV format
                    buyer_email='',  # Not in this CSV format
                    
                    # Shipping info
                    ship_to_name='',
                    ship_to_city='',
                    ship_to_state='',
                    ship_to_zip='',
                    ship_to_country='',
                    
                    # Dates
                    sale_date=str(get_value('sale_date')),
                    paid_on_date='',
                    shipped_on_date='',
                    
                    # Shipping details
                    shipping_service=str(get_value('shipping_service'))[:100],
                    tracking_number=str(get_value('tracking_number'))[:100],
                    
                    # eBay features
                    promoted_listings='',
                    ebay_plus='',
                    global_shipping='',
                    
                    # Additional fields
                    item_location='',
                    item_zip_code='',
                    item_country='',
                    payment_method=str(get_value('payment_method'))[:50],
                    feedback_left='',
                    feedback_received=''
                )
                
                # Calculate initial profit (COGS and shipping cost will be 0)
                order.calculate_profit()
                
                session.add(order)
                orders_created += 1
                
            except ValueError as ve:
                error_msg = f'Row {index + 1}: Validation error - {str(ve)}'
                errors.append(error_msg)
                logger.warning(error_msg)
                continue
            except Exception as row_error:
                error_msg = f'Row {index + 1}: Processing error - {str(row_error)}'
                errors.append(error_msg)
                logger.warning(error_msg)
                continue
        
        # Commit all changes
        session.commit()
        
        logger.info(f"CSV processing complete: {orders_created} created, {orders_updated} updated, {orders_skipped} skipped, {len(errors)} errors")
        
        response_data = {
            'message': 'CSV upload completed',
            'summary': {
                'orders_created': orders_created,
                'orders_updated': orders_updated,
                'orders_skipped': orders_skipped,
                'total_rows_processed': len(csv_data),
                'error_count': len(errors)
            },
            'validation_warnings': validation_warnings,
            'processing_errors': errors[:10] if errors else []  # Limit errors to first 10
        }
        
        if len(errors) > 10:
            response_data['note'] = f'Showing first 10 of {len(errors)} total errors'
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"CSV upload failed: {str(e)}")
        session.rollback()
        return jsonify({'error': f'Failed to process CSV: {str(e)}'}), 500
    finally:
        session.close()

@app.route('/api/export', methods=['GET'])
def export_orders():
    """Export orders as CSV with profit data"""
    session = get_db_session()
    
    try:
        # Get all orders
        orders = session.query(Order).order_by(Order.created_at.desc()).all()
        
        if not orders:
            return jsonify({'error': 'No orders to export'}), 404
        
        # Create CSV data
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write headers
        headers = [
            'Order ID', 'Sales Record Number', 'Order Number', 'Item Title', 'Quantity',
            'Sold For', 'Shipping & Handling', 'eBay Tax', 'Total Price',
            'COGS', 'Shipping Cost', 'Total Profit',
            'Sale Date', 'Buyer', 'Ship To City', 'Ship To State'
        ]
        writer.writerow(headers)
        
        # Write order data
        for order in orders:
            writer.writerow([
                order.id,
                order.sales_record_number,
                order.order_number,
                order.item_title,
                order.quantity,
                f'${order.sold_for:.2f}',
                f'${order.shipping_and_handling:.2f}',
                f'${order.ebay_collected_tax:.2f}',
                f'${order.total_price:.2f}',
                f'${order.cogs:.2f}',
                f'${order.shipping_cost:.2f}',
                f'${order.total_profit:.2f}',
                order.sale_date,
                order.buyer_username,
                order.ship_to_city,
                order.ship_to_state
            ])
        
        # Prepare file for download
        output.seek(0)
        
        # Create a bytes buffer for the file
        csv_buffer = io.BytesIO()
        csv_buffer.write(output.getvalue().encode('utf-8'))
        csv_buffer.seek(0)
        
        filename = f'ebay_orders_with_profit_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        
        return send_file(
            csv_buffer,
            as_attachment=True,
            download_name=filename,
            mimetype='text/csv'
        )
        
    except Exception as e:
        return jsonify({'error': f'Failed to export orders: {str(e)}'}), 500
    finally:
        session.close()

@app.route('/api/debug-orders', methods=['GET'])
def debug_orders():
    """Debug endpoint to see raw order data"""
    session = get_db_session()
    try:
        orders = session.query(Order).limit(5).all()
        debug_info = []
        for order in orders:
            debug_info.append({
                'id': order.id,
                'sales_record_number': order.sales_record_number,
                'order_number': order.order_number,
                'item_title': order.item_title,
                'sold_for': order.sold_for,
                'total_price': order.total_price,
                'sale_date': order.sale_date,
                'buyer_username': order.buyer_username,
                'all_fields': order.to_dict()
            })
        return jsonify({'orders': debug_info, 'total_count': session.query(Order).count()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()

@app.route('/api/clear-orders', methods=['DELETE'])
def clear_orders():
    """Clear all orders from database (for testing)"""
    session = get_db_session()
    try:
        count = session.query(Order).count()
        session.query(Order).delete()
        session.commit()
        logger.info(f"Cleared {count} orders from database")
        return jsonify({'message': f'Cleared {count} orders from database'})
    except Exception as e:
        session.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()

# ========================================
# eBay OAuth 2.0 Authentication Routes
# ========================================

@app.route('/api/auth/ebay/login', methods=['GET'])
def ebay_login():
    """Initiate eBay OAuth flow"""
    try:
        if oauth_service is None:
            return jsonify({'error': 'eBay OAuth service not available. Please check your API credentials.'}), 500
        
        auth_url = oauth_service.generate_user_authorization_url()
        state = "csrf_protection_token"  # In production, generate a random state token
        
        # Store state in session for CSRF protection (in production, use proper session management)
        # For now, we'll return both URL and state to the frontend
        
        logger.info(f"Generated eBay auth URL for state: {state}")
        
        return jsonify({
            'auth_url': auth_url,
            'state': state
        })
        
    except Exception as e:
        logger.error(f"Failed to generate eBay auth URL: {str(e)}")
        return jsonify({'error': 'Failed to generate authorization URL'}), 500

@app.route('/api/auth/ebay/callback', methods=['GET', 'POST'])
def ebay_callback():
    """Handle eBay OAuth callback"""
    try:
        if oauth_service is None:
            return jsonify({'error': 'eBay OAuth service not available. Please check your API credentials.'}), 500
        # Get authorization code from query parameters
        code = request.args.get('code')
        state = request.args.get('state')
        error = request.args.get('error')
        
        if error:
            logger.error(f"eBay OAuth error: {error}")
            return jsonify({'error': f'Authorization failed: {error}'}), 400
        
        if not code:
            logger.error("No authorization code received from eBay")
            return jsonify({'error': 'No authorization code received'}), 400
        
        logger.info(f"Received eBay callback with code: {code[:10]}... and state: {state}")
        
        # Exchange code for tokens
        token_data = oauth_service.exchange_code_for_user_token(code)
        
        # Get seller information
        access_token = token_data['access_token']
        seller_info = oauth_service.get_seller_info(access_token)
        
        # Store tokens and seller info in database
        if oauth_service.store_tokens(token_data, seller_info):
            logger.info(f"Successfully stored eBay tokens for seller: {seller_info.get('sellerId', 'unknown')}")
            
            return jsonify({
                'success': True,
                'message': 'eBay authentication successful',
                'seller_info': {
                    'seller_id': seller_info.get('sellerId'),
                    'scopes': token_data.get('scope', '').split()
                }
            })
        else:
            logger.error("Failed to store eBay tokens in database")
            return jsonify({'error': 'Failed to store authentication data'}), 500
            
    except Exception as e:
        logger.error(f"eBay callback error: {str(e)}")
        return jsonify({'error': f'Authentication failed: {str(e)}'}), 500

@app.route('/api/auth/ebay/status', methods=['GET'])
def ebay_auth_status():
    """Check eBay authentication status"""
    try:
        session = get_db_session()
        
        # Get all active tokens
        tokens = session.query(EBayToken).filter_by(is_active=True).all()
        
        if not tokens:
            return jsonify({
                'authenticated': False,
                'message': 'No active eBay authentication found'
            })
        
        # Get seller information for active tokens
        sellers = []
        for token in tokens:
            seller = session.query(EBaySeller).filter_by(seller_id=token.seller_id).first()
            
            # Check if token is still valid
            is_valid = oauth_service.test_token_validity(token.access_token)
            
            sellers.append({
                'seller_id': token.seller_id,
                'seller_username': seller.seller_username if seller else 'Unknown',
                'token_expires_at': token.token_expires_at.isoformat() if token.token_expires_at else None,
                'scope': token.scope,
                'is_valid': is_valid,
                'marketplace_id': seller.marketplace_id if seller else 'EBAY_US'
            })
        
        return jsonify({
            'authenticated': True,
            'sellers': sellers
        })
        
    except Exception as e:
        logger.error(f"Error checking eBay auth status: {str(e)}")
        return jsonify({'error': 'Failed to check authentication status'}), 500
    finally:
        session.close()

@app.route('/api/auth/ebay/refresh', methods=['POST'])
def refresh_ebay_token():
    """Manually refresh eBay token for a seller"""
    try:
        data = request.get_json()
        seller_id = data.get('seller_id')
        
        if not seller_id:
            return jsonify({'error': 'seller_id is required'}), 400
        
        # Get valid token (this will auto-refresh if needed)
        valid_token = oauth_service.get_valid_token(seller_id)
        
        if valid_token:
            logger.info(f"Successfully refreshed token for seller: {seller_id}")
            return jsonify({
                'success': True,
                'message': 'Token refreshed successfully'
            })
        else:
            logger.warning(f"Failed to refresh token for seller: {seller_id}")
            return jsonify({
                'success': False,
                'message': 'Failed to refresh token - re-authentication may be required'
            }), 401
            
    except Exception as e:
        logger.error(f"Error refreshing eBay token: {str(e)}")
        return jsonify({'error': f'Failed to refresh token: {str(e)}'}), 500

@app.route('/api/auth/ebay/logout', methods=['POST'])
def ebay_logout():
    """Revoke eBay authentication"""
    try:
        data = request.get_json()
        seller_id = data.get('seller_id')
        
        if not seller_id:
            return jsonify({'error': 'seller_id is required'}), 400
        
        session = get_db_session()
        
        # Mark token as inactive
        token = session.query(EBayToken).filter_by(seller_id=seller_id).first()
        if token:
            token.is_active = False
            session.commit()
            logger.info(f"Revoked eBay authentication for seller: {seller_id}")
            
            return jsonify({
                'success': True,
                'message': 'eBay authentication revoked'
            })
        else:
            return jsonify({'error': 'Authentication not found'}), 404
            
    except Exception as e:
        logger.error(f"Error revoking eBay authentication: {str(e)}")
        return jsonify({'error': f'Failed to revoke authentication: {str(e)}'}), 500
    finally:
        session.close()

# ========================================
# New Order Synchronization Routes for Task 4.2
# ========================================

@app.route('/api/orders/sync', methods=['POST'])
def sync_orders():
    """Sync orders from eBay API for specified seller and date range"""
    try:
        data = request.get_json() or {}
        seller_id = data.get('seller_id')
        
        if not seller_id:
            return jsonify({'success': False, 'error': 'Missing seller_id'}), 400
        
        # Optional date range
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        date_range = (start_date, end_date) if start_date and end_date else None
        
        # Initialize sync service
        from ebay_order_sync import EBayOrderSyncService
        sync_service = EBayOrderSyncService('sandbox')
        
        # Perform sync
        result = sync_service.sync_orders(seller_id, date_range)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 400
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/orders/sync/<order_id>', methods=['POST'])
def sync_single_order(order_id):
    """Sync specific order by eBay order ID"""
    try:
        data = request.get_json() or {}
        seller_id = data.get('seller_id')
        
        if not seller_id:
            return jsonify({'success': False, 'error': 'Missing seller_id'}), 400
        
        # Initialize sync service
        from ebay_order_sync import EBayOrderSyncService
        sync_service = EBayOrderSyncService('sandbox')
        
        # Sync single order
        result = sync_service.sync_single_order(seller_id, order_id)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 400
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/orders/sync/status', methods=['GET'])
def get_sync_status():
    """Get synchronization status and statistics"""
    try:
        seller_id = request.args.get('seller_id')
        
        # Initialize sync service
        from ebay_order_sync import EBayOrderSyncService
        sync_service = EBayOrderSyncService('sandbox')
        
        # Get sync status
        status = sync_service.get_sync_status(seller_id)
        
        if 'error' in status:
            return jsonify({'success': False, 'error': status['error']}), 500
        
        return jsonify({'success': True, 'data': status})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/orders/profit/<order_id>', methods=['GET'])
def calculate_order_profit(order_id):
    """Calculate profit metrics for specific order"""
    try:
        seller_id = request.args.get('seller_id')
        
        if not seller_id:
            return jsonify({'success': False, 'error': 'Missing seller_id parameter'}), 400
        
        # Initialize sync service
        from ebay_order_sync import EBayOrderSyncService
        sync_service = EBayOrderSyncService('sandbox')
        
        # Get order data from eBay API (this would typically be cached locally)
        access_token = sync_service.oauth_service.get_valid_token(seller_id)
        if not access_token:
            return jsonify({'success': False, 'error': 'No valid access token'}), 401
        
        # Fetch order data from eBay
        url = f"{sync_service.base_url}/sell/fulfillment/{sync_service.API_VERSION}/order/{order_id}"
        headers = sync_service.get_headers(access_token)
        
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            return jsonify({'success': False, 'error': f'eBay API error: {response.status_code}'}), 400
        
        order_data = response.json()
        profit_metrics = sync_service.calculate_profit_metrics(order_data)
        
        return jsonify({'success': True, 'data': profit_metrics})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ========================================
# New Dashboard API Endpoints for Task 4.3
# ========================================

@app.route('/api/dashboard/metrics', methods=['GET'])
def get_dashboard_metrics():
    """Get comprehensive dashboard metrics and analytics data"""
    try:
        session = get_db_session()
        
        # Get date filters
        today = datetime.now().date()
        thirty_days_ago = today - timedelta(days=30)
        seven_days_ago = today - timedelta(days=7)
        
        # Calculate core metrics
        total_orders = session.query(Order).count()
        
        # Today's metrics (using string comparison for sale_date)
        today_str = today.strftime('%Y-%m-%d')
        todays_orders = session.query(Order).filter(
            Order.sale_date.like(f'{today_str}%')
        ).all()
        
        todaysSales = sum(order.total_price or 0 for order in todays_orders)
        todaysProfit = sum(order.total_profit or 0 for order in todays_orders)
        
        # Overall metrics
        all_orders = session.query(Order).all()
        totalRevenue = sum(order.total_price or 0 for order in all_orders)
        totalProfit = sum(order.total_profit or 0 for order in all_orders)
        totalCogs = sum(order.cogs or 0 for order in all_orders)
        
        averageOrderValue = totalRevenue / total_orders if total_orders > 0 else 0
        profitMargin = (totalProfit / totalRevenue * 100) if totalRevenue > 0 else 0
        
        # Sync status
        synced_orders = session.query(Order).filter(Order.sync_status == 'SYNCED').count()
        syncPercentage = (synced_orders / total_orders * 100) if total_orders > 0 else 0
        
        metrics = {
            'todaysSales': todaysSales,
            'todaysProfit': todaysProfit,
            'orderCount': total_orders,
            'averageOrderValue': averageOrderValue,
            'profitMargin': profitMargin,
            'syncPercentage': syncPercentage,
            'totalRevenue': totalRevenue,
            'totalCogs': totalCogs
        }
        
        # Generate chart data (last 7 days)
        chartData = []
        for i in range(7):
            date = today - timedelta(days=6-i)
            date_str = date.strftime('%Y-%m-%d')
            
            daily_orders = session.query(Order).filter(
                Order.sale_date.like(f'{date_str}%')
            ).all()
            
            daily_sales = sum(order.total_price or 0 for order in daily_orders)
            daily_profit = sum(order.total_profit or 0 for order in daily_orders)
            
            chartData.append({
                'date': date.strftime('%m/%d'),
                'sales': daily_sales,
                'profit': daily_profit,
                'orders': len(daily_orders)
            })
        
        # Get top products (by profit)
        topProducts = []
        product_stats = {}
        
        for order in all_orders:
            if order.item_title:
                title = order.item_title
                if title not in product_stats:
                    product_stats[title] = {
                        'name': title,
                        'quantity': 0,
                        'revenue': 0,
                        'profit': 0
                    }
                
                product_stats[title]['quantity'] += order.quantity or 1
                product_stats[title]['revenue'] += order.total_price or 0
                product_stats[title]['profit'] += order.total_profit or 0
        
        # Sort by profit and take top 5
        sorted_products = sorted(product_stats.values(), key=lambda x: x['profit'], reverse=True)
        topProducts = sorted_products[:5]
        
        # Get recent orders (last 10)
        recent_orders = session.query(Order).order_by(desc(Order.created_at)).limit(10).all()
        recentOrders = []
        
        for order in recent_orders:
            recentOrders.append({
                'id': order.id,
                'title': order.item_title or 'Unknown Item',
                'buyer': order.buyer_username or 'Unknown Buyer',
                'amount': order.total_price or 0,
                'profit': order.total_profit or 0,
                'date': order.sale_date or 'Unknown',
                'status': order.sync_status or 'Manual'
            })
        
        session.close()
        
        return jsonify({
            'success': True,
            'metrics': metrics,
            'chartData': chartData,
            'topProducts': topProducts,
            'recentOrders': recentOrders
        })
        
    except Exception as e:
        logger.error(f"Dashboard metrics error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/dashboard/analytics', methods=['GET'])
def get_dashboard_analytics():
    """Get advanced analytics data for dashboard"""
    try:
        session = get_db_session()
        
        # Get time range from query params
        days = request.args.get('days', '30', type=int)
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        
        # Profit trend analysis
        profit_trend = []
        for i in range(days):
            date = start_date + timedelta(days=i)
            date_str = date.strftime('%Y-%m-%d')
            
            daily_orders = session.query(Order).filter(
                Order.sale_date.like(f'{date_str}%')
            ).all()
            
            daily_profit = sum(order.total_profit or 0 for order in daily_orders)
            profit_trend.append({
                'date': date.strftime('%Y-%m-%d'),
                'profit': daily_profit,
                'orders': len(daily_orders)
            })
        
        # Category analysis (simplified - group by first word of item title)
        category_stats = {}
        all_orders = session.query(Order).all()
        
        for order in all_orders:
            if order.item_title:
                # Extract first word as category
                category = order.item_title.split()[0] if order.item_title.split() else 'Other'
                if category not in category_stats:
                    category_stats[category] = {
                        'name': category,
                        'orders': 0,
                        'revenue': 0,
                        'profit': 0
                    }
                
                category_stats[category]['orders'] += 1
                category_stats[category]['revenue'] += order.total_price or 0
                category_stats[category]['profit'] += order.total_profit or 0
        
        # Sort categories by revenue
        categories = sorted(category_stats.values(), key=lambda x: x['revenue'], reverse=True)[:10]
        
        session.close()
        
        return jsonify({
            'success': True,
            'profitTrend': profit_trend,
            'categories': categories,
            'timeRange': f'{start_date} to {end_date}'
        })
        
    except Exception as e:
        logger.error(f"Dashboard analytics error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/dashboard/export', methods=['GET'])
def export_dashboard_data():
    """Export dashboard data as CSV"""
    try:
        session = get_db_session()
        
        # Get export type
        export_type = request.args.get('type', 'summary')
        
        if export_type == 'summary':
            # Export summary metrics
            orders = session.query(Order).all()
            
            summary_data = {
                'total_orders': len(orders),
                'total_revenue': sum(order.total_price or 0 for order in orders),
                'total_profit': sum(order.total_profit or 0 for order in orders),
                'total_cogs': sum(order.cogs or 0 for order in orders),
                'average_order_value': sum(order.total_price or 0 for order in orders) / len(orders) if orders else 0,
                'profit_margin': (sum(order.total_profit or 0 for order in orders) / sum(order.total_price or 0 for order in orders) * 100) if sum(order.total_price or 0 for order in orders) > 0 else 0
            }
            
            session.close()
            
            return jsonify({
                'success': True,
                'data': summary_data,
                'export_type': 'summary'
            })
        
        else:
            # Export detailed orders
            orders = session.query(Order).limit(1000).all()  # Limit for performance
            
            orders_data = []
            for order in orders:
                orders_data.append({
                    'id': order.id,
                    'item_title': order.item_title,
                    'sale_date': order.sale_date,
                    'total_price': order.total_price,
                    'total_profit': order.total_profit,
                    'buyer_username': order.buyer_username,
                    'sync_status': order.sync_status
                })
            
            session.close()
            
            return jsonify({
                'success': True,
                'data': orders_data,
                'export_type': 'detailed'
            })
        
    except Exception as e:
        logger.error(f"Dashboard export error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/dashboard/sync-trigger', methods=['POST'])
def trigger_dashboard_sync():
    """Trigger order synchronization from dashboard"""
    try:
        data = request.get_json() or {}
        seller_id = data.get('seller_id', 'default_seller')
        
        # Import sync service
        from ebay_order_sync import EBayOrderSyncService
        sync_service = EBayOrderSyncService('sandbox')
        
        # Perform sync
        result = sync_service.sync_orders(seller_id)
        
        return jsonify({
            'success': True,
            'sync_result': result,
            'message': 'Sync triggered successfully'
        })
        
    except Exception as e:
        logger.error(f"Dashboard sync trigger error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

# ========================================
# INVENTORY MANAGEMENT ENDPOINTS
# ========================================

@app.route('/api/inventory/items', methods=['GET'])
def get_inventory_items():
    """Get all inventory items with optional filtering"""
    try:
        # Get query parameters
        seller_id = request.args.get('seller_id', 'default_seller')
        search = request.args.get('search', '')
        limit = int(request.args.get('limit', 50))
        offset = int(request.args.get('offset', 0))
        low_stock_only = request.args.get('low_stock_only', 'false').lower() == 'true'
        
        conn = sqlite3.connect('ebay_seller_tools.db')
        cursor = conn.cursor()
        
        # Build query with filters
        base_query = '''
            SELECT 
                ii.item_id, ii.sku, ii.title, ii.condition_id, ii.brand,
                ii.package_weight_oz, ii.package_dimensions,
                iq.available_quantity, iq.total_quantity, iq.sync_status,
                iq.last_sync_at, ii.created_at, ii.updated_at
            FROM inventory_items ii
            LEFT JOIN inventory_quantities iq ON ii.item_id = iq.item_id
            WHERE ii.seller_id = ?
        '''
        
        query_params = [seller_id]
        
        # Add search filter
        if search:
            base_query += ' AND (ii.title LIKE ? OR ii.sku LIKE ? OR ii.brand LIKE ?)'
            search_param = f'%{search}%'
            query_params.extend([search_param, search_param, search_param])
        
        # Add low stock filter
        if low_stock_only:
            base_query += ' AND iq.available_quantity <= 5'
        
        # Add ordering and pagination
        base_query += ' ORDER BY ii.updated_at DESC LIMIT ? OFFSET ?'
        query_params.extend([limit, offset])
        
        cursor.execute(base_query, query_params)
        items = cursor.fetchall()
        
        # Get total count for pagination
        count_query = base_query.replace(
            'SELECT \n                ii.item_id, ii.sku, ii.title, ii.condition_id, ii.brand,\n                ii.package_weight_oz, ii.package_dimensions,\n                iq.available_quantity, iq.total_quantity, iq.sync_status,\n                iq.last_sync_at, ii.created_at, ii.updated_at', 
            'SELECT COUNT(*)'
        )
        count_query = count_query.replace(' ORDER BY ii.updated_at DESC LIMIT ? OFFSET ?', '')
        cursor.execute(count_query, query_params[:-2])
        count_result = cursor.fetchone()
        total_count = count_result[0] if count_result else 0
        
        conn.close()
        
        # Format results
        inventory_items = []
        for item in items:
            inventory_items.append({
                'item_id': item[0],
                'sku': item[1],
                'title': item[2],
                'condition_id': item[3],
                'brand': item[4],
                'package_weight_oz': item[5],
                'package_dimensions': item[6],
                'available_quantity': item[7] or 0,
                'total_quantity': item[8] or 0,
                'sync_status': item[9] or 'PENDING',
                'last_sync_at': item[10],
                'created_at': item[11],
                'updated_at': item[12]
            })
        
        return jsonify({
            'status': 'success',
            'inventory_items': inventory_items,
            'pagination': {
                'total_count': total_count,
                'limit': limit,
                'offset': offset,
                'has_more': (offset + limit) < total_count
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting inventory items: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'inventory_items': []
        }), 500

@app.route('/api/inventory/sync', methods=['POST'])
def sync_inventory_from_ebay():
    """Sync inventory items from eBay Inventory API"""
    try:
        if not ebay_inventory_service:
            logger.warning("eBay inventory service not available")
            return jsonify({
                'status': 'error',
                'error': 'eBay inventory service not available',
                'items_synced': 0,
                'offers_synced': 0
            }), 503
        
        # Check if eBay service is properly configured
        if not ebay_inventory_service.is_available():
            logger.info("eBay service not configured, returning demo data message")
            return jsonify({
                'status': 'demo_mode',
                'error': 'eBay API credentials not configured - running in demo mode',
                'message': 'To enable live eBay integration, configure your eBay Developer API credentials',
                'authorization_url': ebay_inventory_service.get_user_authorization_url(),
                'items_synced': 0,
                'offers_synced': 0
            }), 200
        
        # Perform actual eBay sync
        logger.info("Starting eBay inventory sync...")
        sync_results = ebay_inventory_service.sync_inventory_from_ebay()
        
        if sync_results['success']:
            logger.info(f"eBay sync successful: {sync_results['items_synced']} items, {sync_results['offers_synced']} offers")
            return jsonify(sync_results)
        else:
            logger.error(f"eBay sync failed: {sync_results.get('error', 'Unknown error')}")
            return jsonify(sync_results), 500
        
    except Exception as e:
        logger.error(f"Error syncing inventory from eBay: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'items_synced': 0,
            'offers_synced': 0
        }), 500

@app.route('/api/ebay/status', methods=['GET'])
def get_ebay_service_status():
    """Get eBay service status and configuration"""
    try:
        if not ebay_inventory_service:
            return jsonify({
                'service_available': False,
                'error': 'eBay service not loaded'
            }), 503
        
        status = ebay_inventory_service.get_service_status()
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"Error getting eBay service status: {e}")
        return jsonify({
            'service_available': False,
            'error': str(e)
        }), 500

@app.route('/api/ebay/auth/url', methods=['GET'])
def get_ebay_auth_url():
    """Get eBay user authorization URL"""
    try:
        if not ebay_inventory_service:
            return jsonify({'error': 'eBay service not available'}), 503
        
        auth_url = ebay_inventory_service.get_user_authorization_url()
        
        if auth_url:
            return jsonify({
                'authorization_url': auth_url,
                'instructions': 'Visit this URL to authorize the application with your eBay account'
            })
        else:
            return jsonify({
                'error': 'Cannot generate authorization URL - eBay credentials not configured'
            }), 400
            
    except Exception as e:
        logger.error(f"Error generating eBay auth URL: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ebay/auth/callback', methods=['POST'])
def handle_ebay_auth_callback():
    """Handle eBay authorization callback"""
    try:
        if not ebay_inventory_service:
            return jsonify({'error': 'eBay service not available'}), 503
        
        data = request.get_json() or {}
        auth_code = data.get('code')
        
        if not auth_code:
            return jsonify({'error': 'Authorization code required'}), 400
        
        result = ebay_inventory_service.exchange_authorization_code(auth_code)
        
        if result['success']:
            logger.info("eBay authorization successful")
            return jsonify(result)
        else:
            logger.error(f"eBay authorization failed: {result['error']}")
            return jsonify(result), 400
            
    except Exception as e:
        logger.error(f"Error handling eBay auth callback: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/inventory/quantities/sync', methods=['POST'])
def sync_inventory_quantities():
    """Sync current stock levels from eBay"""
    try:
        if not ebay_inventory_service:
            return jsonify({
                'status': 'error',
                'error': 'eBay inventory service not available',
                'synced_quantities': 0
            }), 503
        
        if not ebay_inventory_service.is_available():
            return jsonify({
                'status': 'demo_mode',
                'error': 'eBay API credentials not configured - running in demo mode',
                'synced_quantities': 0
            }), 200
        
        data = request.get_json() or {}
        limit = data.get('limit', 100)
        
        # Get current inventory from eBay
        logger.info("Syncing inventory quantities from eBay...")
        items_response = ebay_inventory_service.get_inventory_items(limit=limit)
        
        if 'error' in items_response:
            logger.error(f"Failed to sync quantities: {items_response['error']}")
            return jsonify({
                'status': 'error',
                'error': items_response['error'],
                'synced_quantities': 0
            }), 500
        
        items = items_response.get('inventoryItems', [])
        synced_count = len(items)
        
        logger.info(f"Successfully synced {synced_count} inventory quantities")
        return jsonify({
            'status': 'success',
            'synced_quantities': synced_count,
            'total_items': len(items),
            'message': f'Synced quantities for {synced_count} items from eBay'
        })
        
    except Exception as e:
        logger.error(f"Error syncing inventory quantities: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'synced_quantities': 0
        }), 500

@app.route('/api/inventory/locations', methods=['GET'])
def get_inventory_locations():
    """Get all inventory locations/warehouses"""
    try:
        seller_id = request.args.get('seller_id', 'default_seller')
        
        conn = sqlite3.connect('ebay_seller_tools.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT location_id, location_name, location_type, address_line1, 
                   address_line2, city, state_province, postal_code, country_code,
                   is_active, is_default, created_at, updated_at
            FROM inventory_locations
            WHERE seller_id = ?
            ORDER BY is_default DESC, location_name ASC
        ''', (seller_id,))
        
        locations = cursor.fetchall()
        conn.close()
        
        inventory_locations = []
        for location in locations:
            inventory_locations.append({
                'location_id': location[0],
                'location_name': location[1],
                'location_type': location[2],
                'address_line1': location[3],
                'address_line2': location[4],
                'city': location[5],
                'state_province': location[6],
                'postal_code': location[7],
                'country_code': location[8],
                'is_active': bool(location[9]),
                'is_default': bool(location[10]),
                'created_at': location[11],
                'updated_at': location[12]
            })
        
        return jsonify({
            'status': 'success',
            'locations': inventory_locations
        })
        
    except Exception as e:
        logger.error(f"Error getting inventory locations: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'locations': []
        }), 500

@app.route('/api/inventory/valuation', methods=['GET'])
def get_inventory_valuation():
    """Get current inventory valuation and metrics"""
    try:
        seller_id = request.args.get('seller_id', 'default_seller')
        
        # Get inventory valuation directly from database
        conn = sqlite3.connect('ebay_seller_tools.db')
        cursor = conn.cursor()
        
        # Get inventory valuation
        cursor.execute('''
            SELECT 
                ii.sku,
                ii.title,
                iq.available_quantity,
                iq.total_quantity,
                AVG(ic.unit_cost) as avg_unit_cost,
                SUM(ic.unit_cost * ic.quantity_purchased) / SUM(ic.quantity_purchased) as weighted_avg_cost
            FROM inventory_items ii
            LEFT JOIN inventory_quantities iq ON ii.item_id = iq.item_id
            LEFT JOIN inventory_costs ic ON ii.item_id = ic.item_id AND ic.is_active = TRUE
            WHERE ii.seller_id = ?
            GROUP BY ii.item_id, ii.sku, ii.title, iq.available_quantity, iq.total_quantity
            HAVING iq.available_quantity > 0
        ''', (seller_id,))
        
        items = cursor.fetchall()
        
        total_value = 0
        total_quantity = 0
        
        inventory_items = []
        for item in items:
            sku, title, available_qty, total_qty, avg_cost, weighted_cost = item
            unit_cost = weighted_cost or avg_cost or 0
            item_value = (available_qty or 0) * unit_cost
            
            inventory_items.append({
                'sku': sku,
                'title': title,
                'available_quantity': available_qty or 0,
                'total_quantity': total_qty or 0,
                'unit_cost': round(unit_cost, 4) if unit_cost else 0,
                'total_value': round(item_value, 2)
            })
            
            total_value += item_value
            total_quantity += (available_qty or 0)
        
        result = {
            'status': 'success',
            'total_value': round(total_value, 2),
            'total_quantity': total_quantity,
            'item_count': len(inventory_items),
            'inventory_items': inventory_items
        }
        
        # Add additional metrics
        conn = sqlite3.connect('ebay_seller_tools.db')
        cursor = conn.cursor()
        
        # Get low stock alerts
        cursor.execute('''
            SELECT COUNT(*) as low_stock_count
            FROM inventory_quantities iq
            JOIN inventory_items ii ON iq.item_id = ii.item_id
            JOIN reorder_rules rr ON ii.item_id = rr.item_id
            WHERE ii.seller_id = ? 
            AND iq.available_quantity <= rr.reorder_point 
            AND rr.is_active = TRUE
        ''', (seller_id,))
        
        low_stock_result = cursor.fetchone()
        low_stock_count = low_stock_result[0] if low_stock_result else 0
        
        # Get out of stock count
        cursor.execute('''
            SELECT COUNT(*) as out_of_stock_count
            FROM inventory_quantities iq
            JOIN inventory_items ii ON iq.item_id = ii.item_id
            WHERE ii.seller_id = ? AND iq.available_quantity = 0
        ''', (seller_id,))
        
        out_of_stock_result = cursor.fetchone()
        out_of_stock_count = out_of_stock_result[0] if out_of_stock_result else 0
        
        conn.close()
        
        # Add metrics to result
        result['metrics'] = {
            'low_stock_alerts': low_stock_count,
            'out_of_stock_count': out_of_stock_count,
            'total_locations': 1,  # Will be dynamic when multiple locations are supported
            'sync_status': 'ACTIVE'
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error getting inventory valuation: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'total_value': 0,
            'metrics': {}
        }), 500

@app.route('/api/inventory/costs', methods=['POST'])
def add_inventory_cost():
    """Add a cost entry for inventory tracking (COGS)"""
    try:
        data = request.get_json()
        
        required_fields = ['sku', 'unit_cost', 'quantity_purchased', 'purchase_date']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'status': 'error',
                    'error': f'Missing required field: {field}'
                }), 400
        
        conn = sqlite3.connect('ebay_seller_tools.db')
        cursor = conn.cursor()
        
        # Get item_id from SKU
        cursor.execute('SELECT item_id FROM inventory_items WHERE sku = ?', (data['sku'],))
        result = cursor.fetchone()
        
        if not result:
            conn.close()
            return jsonify({
                'status': 'error',
                'error': f'Item with SKU {data["sku"]} not found'
            }), 404
        
        item_id = result[0]
        
        # Calculate total landed cost
        unit_cost = float(data['unit_cost'])
        quantity = int(data['quantity_purchased'])
        shipping_cost_per_unit = float(data.get('shipping_cost_per_unit', 0))
        total_landed_cost = (unit_cost + shipping_cost_per_unit) * quantity
        
        # Insert cost record
        cursor.execute('''
            INSERT INTO inventory_costs 
            (item_id, purchase_date, supplier_name, purchase_order_number, 
             unit_cost, quantity_purchased, shipping_cost_per_unit, 
             total_landed_cost, currency, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            item_id, data['purchase_date'], data.get('supplier_name'),
            data.get('purchase_order_number'), unit_cost, quantity,
            shipping_cost_per_unit, total_landed_cost,
            data.get('currency', 'USD'), data.get('notes')
        ))
        
        cost_id = cursor.lastrowid
        
        # Update inventory quantity if this is new stock
        if data.get('add_to_inventory', False):
            cursor.execute('''
                UPDATE inventory_quantities 
                SET available_quantity = available_quantity + ?,
                    total_quantity = total_quantity + ?,
                    updated_at = ?
                WHERE item_id = ? AND location_id = ?
            ''', (
                quantity, quantity, datetime.now().isoformat(),
                item_id, data.get('location_id', 'DEFAULT_WAREHOUSE')
            ))
            
            # Record inventory movement
            cursor.execute('''
                INSERT INTO inventory_movements
                (item_id, location_id, movement_type, quantity_change, 
                 reference_type, reference_id, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                item_id, data.get('location_id', 'DEFAULT_WAREHOUSE'),
                'PURCHASE', quantity, 'COST_ENTRY', cost_id,
                f'Purchase from {data.get("supplier_name", "Unknown Supplier")}'
            ))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'status': 'success',
            'cost_id': cost_id,
            'message': 'Cost entry added successfully'
        })
        
    except Exception as e:
        logger.error(f"Error adding inventory cost: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/inventory/reorder-rules', methods=['GET', 'POST'])
def manage_reorder_rules():
    """Get or create reorder rules for inventory management"""
    try:
        if request.method == 'GET':
            seller_id = request.args.get('seller_id', 'default_seller')
            
            conn = sqlite3.connect('ebay_seller_tools.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT 
                    rr.rule_id, ii.sku, ii.title, rr.location_id,
                    rr.reorder_point, rr.reorder_quantity, rr.max_stock_level,
                    rr.is_active, rr.alert_enabled, rr.supplier_lead_time_days,
                    rr.preferred_supplier, iq.available_quantity
                FROM reorder_rules rr
                JOIN inventory_items ii ON rr.item_id = ii.item_id
                LEFT JOIN inventory_quantities iq ON rr.item_id = iq.item_id AND rr.location_id = iq.location_id
                WHERE ii.seller_id = ?
                ORDER BY ii.title ASC
            ''', (seller_id,))
            
            rules = cursor.fetchall()
            conn.close()
            
            reorder_rules = []
            for rule in rules:
                reorder_rules.append({
                    'rule_id': rule[0],
                    'sku': rule[1],
                    'title': rule[2],
                    'location_id': rule[3],
                    'reorder_point': rule[4],
                    'reorder_quantity': rule[5],
                    'max_stock_level': rule[6],
                    'is_active': bool(rule[7]),
                    'alert_enabled': bool(rule[8]),
                    'supplier_lead_time_days': rule[9],
                    'preferred_supplier': rule[10],
                    'current_quantity': rule[11] or 0,
                    'needs_reorder': (rule[11] or 0) <= rule[4]
                })
            
            return jsonify({
                'status': 'success',
                'reorder_rules': reorder_rules
            })
        
        elif request.method == 'POST':
            data = request.get_json()
            
            required_fields = ['sku', 'reorder_point', 'reorder_quantity']
            for field in required_fields:
                if field not in data:
                    return jsonify({
                        'status': 'error',
                        'error': f'Missing required field: {field}'
                    }), 400
            
            conn = sqlite3.connect('ebay_seller_tools.db')
            cursor = conn.cursor()
            
            # Get item_id from SKU
            cursor.execute('SELECT item_id FROM inventory_items WHERE sku = ?', (data['sku'],))
            result = cursor.fetchone()
            
            if not result:
                conn.close()
                return jsonify({
                    'status': 'error',
                    'error': f'Item with SKU {data["sku"]} not found'
                }), 404
            
            item_id = result[0]
            
            # Insert or update reorder rule
            cursor.execute('''
                INSERT OR REPLACE INTO reorder_rules
                (item_id, location_id, reorder_point, reorder_quantity, 
                 max_stock_level, is_active, alert_enabled, supplier_lead_time_days,
                 preferred_supplier, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                item_id, data.get('location_id', 'DEFAULT_WAREHOUSE'),
                int(data['reorder_point']), int(data['reorder_quantity']),
                data.get('max_stock_level'), data.get('is_active', True),
                data.get('alert_enabled', True), data.get('supplier_lead_time_days', 7),
                data.get('preferred_supplier'), datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
            return jsonify({
                'status': 'success',
                'message': 'Reorder rule created/updated successfully'
            })
        
    except Exception as e:
        logger.error(f"Error managing reorder rules: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/inventory/movements', methods=['GET', 'POST'])
def manage_inventory_movements():
    """Get inventory movement history or record new movements"""
    try:
        if request.method == 'GET':
            sku = request.args.get('sku')
            limit = int(request.args.get('limit', 50))
            offset = int(request.args.get('offset', 0))
            
            conn = sqlite3.connect('ebay_seller_tools.db')
            cursor = conn.cursor()
            
            if sku:
                # Get movements for specific SKU
                cursor.execute('''
                    SELECT 
                        im.movement_id, ii.sku, ii.title, im.location_id,
                        im.movement_type, im.quantity_change, im.reference_type,
                        im.reference_id, im.movement_date, im.notes, im.created_by
                    FROM inventory_movements im
                    JOIN inventory_items ii ON im.item_id = ii.item_id
                    WHERE ii.sku = ?
                    ORDER BY im.movement_date DESC
                    LIMIT ? OFFSET ?
                ''', (sku, limit, offset))
            else:
                # Get all recent movements
                cursor.execute('''
                    SELECT 
                        im.movement_id, ii.sku, ii.title, im.location_id,
                        im.movement_type, im.quantity_change, im.reference_type,
                        im.reference_id, im.movement_date, im.notes, im.created_by
                    FROM inventory_movements im
                    JOIN inventory_items ii ON im.item_id = ii.item_id
                    ORDER BY im.movement_date DESC
                    LIMIT ? OFFSET ?
                ''', (limit, offset))
            
            movements = cursor.fetchall()
            conn.close()
            
            inventory_movements = []
            for movement in movements:
                inventory_movements.append({
                    'movement_id': movement[0],
                    'sku': movement[1],
                    'title': movement[2],
                    'location_id': movement[3],
                    'movement_type': movement[4],
                    'quantity_change': movement[5],
                    'reference_type': movement[6],
                    'reference_id': movement[7],
                    'movement_date': movement[8],
                    'notes': movement[9],
                    'created_by': movement[10]
                })
            
            return jsonify({
                'status': 'success',
                'movements': inventory_movements
            })
        
        elif request.method == 'POST':
            data = request.get_json()
            
            required_fields = ['sku', 'movement_type', 'quantity_change']
            for field in required_fields:
                if field not in data:
                    return jsonify({
                        'status': 'error',
                        'error': f'Missing required field: {field}'
                    }), 400
            
            conn = sqlite3.connect('ebay_seller_tools.db')
            cursor = conn.cursor()
            
            # Get item_id from SKU
            cursor.execute('SELECT item_id FROM inventory_items WHERE sku = ?', (data['sku'],))
            result = cursor.fetchone()
            
            if not result:
                conn.close()
                return jsonify({
                    'status': 'error',
                    'error': f'Item with SKU {data["sku"]} not found'
                }), 404
            
            item_id = result[0]
            quantity_change = int(data['quantity_change'])
            location_id = data.get('location_id', 'DEFAULT_WAREHOUSE')
            
            # Record movement
            cursor.execute('''
                INSERT INTO inventory_movements
                (item_id, location_id, movement_type, quantity_change,
                 reference_type, reference_id, notes, created_by)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                item_id, location_id, data['movement_type'], quantity_change,
                data.get('reference_type'), data.get('reference_id'),
                data.get('notes'), data.get('created_by', 'API')
            ))
            
            # Update inventory quantity based on movement type
            if data['movement_type'] in ['SALE', 'DAMAGE', 'LOSS', 'RETURN_OUT']:
                # Decrease quantity
                cursor.execute('''
                    UPDATE inventory_quantities 
                    SET available_quantity = available_quantity - ?,
                        total_quantity = total_quantity - ?,
                        updated_at = ?
                    WHERE item_id = ? AND location_id = ?
                ''', (
                    abs(quantity_change), abs(quantity_change),
                    datetime.now().isoformat(), item_id, location_id
                ))
            elif data['movement_type'] in ['PURCHASE', 'RETURN_IN', 'ADJUSTMENT_UP']:
                # Increase quantity
                cursor.execute('''
                    UPDATE inventory_quantities 
                    SET available_quantity = available_quantity + ?,
                        total_quantity = total_quantity + ?,
                        updated_at = ?
                    WHERE item_id = ? AND location_id = ?
                ''', (
                    abs(quantity_change), abs(quantity_change),
                    datetime.now().isoformat(), item_id, location_id
                ))
            
            conn.commit()
            conn.close()
            
            return jsonify({
                'status': 'success',
                'message': 'Inventory movement recorded successfully'
            })
        
    except Exception as e:
        logger.error(f"Error managing inventory movements: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

# ========================================
# Home Page and Privacy Policy Routes
# ========================================

@app.route('/', methods=['GET'])
def home():
    """Home page for eBay Seller Tools"""
    home_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>eBay Seller Tools</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                max-width: 1200px; 
                margin: 0 auto; 
                padding: 20px; 
                background-color: #f8f9fa;
            }
            .header { 
                text-align: center; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white; 
                padding: 40px; 
                border-radius: 10px; 
                margin-bottom: 30px;
            }
            .header h1 { margin: 0; font-size: 2.5em; }
            .header p { margin: 10px 0 0 0; font-size: 1.2em; opacity: 0.9; }
            .features { 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
                gap: 20px; 
                margin-bottom: 30px;
            }
            .feature { 
                background: white; 
                padding: 25px; 
                border-radius: 8px; 
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .feature h3 { color: #2c3e50; margin-top: 0; }
            .status { 
                text-align: center; 
                background: white; 
                padding: 20px; 
                border-radius: 8px; 
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .btn { 
                display: inline-block; 
                background: #667eea; 
                color: white; 
                padding: 12px 24px; 
                text-decoration: none; 
                border-radius: 5px; 
                margin: 10px;
            }
            .btn:hover { background: #5a6fd8; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🚀 eBay Seller Tools</h1>
            <p>Professional inventory management and profit tracking for eBay sellers</p>
        </div>
        
        <div class="features">
            <div class="feature">
                <h3>📊 Order Management</h3>
                <p>Upload and manage your CSV order files with automatic profit calculations and inventory tracking.</p>
            </div>
            <div class="feature">
                <h3>🔗 eBay Integration</h3>
                <p>Connect directly to eBay's API for real-time inventory synchronization and order data.</p>
            </div>
            <div class="feature">
                <h3>💰 Profit Analytics</h3>
                <p>Track your profit margins, inventory costs, and business performance with detailed reports.</p>
            </div>
        </div>
        
        <div class="status">
            <h3>🔧 System Status</h3>
            <p>Application is running and ready to use!</p>
            <a href="/api/health" class="btn">Check API Status</a>
            <a href="/api/auth/ebay/login" class="btn">Connect eBay Account</a>
        </div>
    </body>
    </html>
    """
    return home_html

@app.route('/privacy', methods=['GET'])
def privacy_policy():
    """Privacy policy page for eBay OAuth compliance"""
    privacy_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Privacy Policy - eBay Seller Tools</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            h1 { color: #2c3e50; }
            h2 { color: #34495e; margin-top: 30px; }
            p { line-height: 1.6; margin-bottom: 15px; }
        </style>
    </head>
    <body>
        <h1>Privacy Policy</h1>
        <p><strong>Effective Date:</strong> December 2024</p>
        
        <h2>Information We Collect</h2>
        <p>Our eBay Seller Tools application collects and processes the following information:</p>
        <ul>
            <li>eBay account information and order data through authorized API access</li>
            <li>Sales records and transaction data from your eBay store</li>
            <li>Inventory and cost data you provide for profit calculations</li>
        </ul>
        
        <h2>How We Use Your Information</h2>
        <p>We use your information solely for:</p>
        <ul>
            <li>Providing order management and profit analysis tools</li>
            <li>Synchronizing your eBay sales data</li>
            <li>Calculating profit margins and business metrics</li>
        </ul>
        
        <h2>Data Security</h2>
        <p>We implement appropriate security measures to protect your data. Your eBay authentication tokens are securely stored and used only for authorized API access.</p>
        
        <h2>Data Sharing</h2>
        <p>We do not sell, trade, or share your personal information with third parties except as required by law.</p>
        
        <h2>Your Rights</h2>
        <p>You may revoke access to your eBay data at any time through your eBay account settings or by contacting us.</p>
        
        <h2>Contact Us</h2>
        <p>For questions about this privacy policy, please contact us through GitHub.</p>
        
        <p><em>This application is designed for seller productivity and operates in compliance with eBay's API policies.</em></p>
    </body>
    </html>
    """
    return privacy_html

# ========================================
# Existing Routes
# ========================================

if __name__ == '__main__':
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    port = int(os.getenv('PORT', 5000))
    host = os.getenv('HOST', '0.0.0.0')  # Railway needs 0.0.0.0, not 127.0.0.1
    app.run(debug=debug_mode, host=host, port=port) 
