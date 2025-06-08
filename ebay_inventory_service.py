#!/usr/bin/env python3
"""
eBay Inventory API Service
Fetches real inventory data from eBay's Inventory API
Integrates with EbayOAuthService for authentication
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
import json
import requests
from ebay_oauth_service import EbayOAuthService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EbayInventoryService:
    """
    Service for interacting with eBay's Inventory API
    Handles fetching inventory items, offers, and locations
    """
    
    def __init__(self, environment="sandbox"):
        """
        Initialize eBay Inventory Service
        
        Args:
            environment (str): 'sandbox' or 'production'
        """
        self.oauth_service = EbayOAuthService(environment)
        self.environment = environment
        
        # eBay API base URLs
        if environment == "sandbox":
            self.inventory_base_url = "https://api.sandbox.ebay.com/sell/inventory/v1"
            self.fulfillment_base_url = "https://api.sandbox.ebay.com/sell/fulfillment/v1"
        else:
            self.inventory_base_url = "https://api.ebay.com/sell/inventory/v1"
            self.fulfillment_base_url = "https://api.ebay.com/sell/fulfillment/v1"
    
    def is_available(self) -> bool:
        """
        Check if eBay API is available and credentials are configured
        
        Returns:
            bool: True if service is available
        """
        token_status = self.oauth_service.get_token_status()
        return token_status['credentials_configured']
    
    def get_inventory_items(self, limit: int = 50, offset: int = 0) -> Dict:
        """
        Get inventory items from eBay
        
        Args:
            limit (int): Number of items to retrieve (max 100)
            offset (int): Offset for pagination
            
        Returns:
            dict: eBay API response with inventory items
        """
        
        if not self.is_available():
            logger.warning("eBay service not available - no credentials configured")
            return {"inventoryItems": [], "total": 0, "error": "Service not available"}
        
        # Build request parameters
        params = {
            'limit': min(limit, 100),  # eBay max is 100
            'offset': offset
        }
        
        try:
            # Make authenticated request
            response = self.oauth_service.make_authenticated_request(
                method='GET',
                endpoint='/inventory_item',
                token_type='user',
                params=params
            )
            
            if response is None:
                logger.error("Failed to make authenticated request to eBay")
                return {"inventoryItems": [], "total": 0, "error": "Authentication failed"}
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Successfully retrieved {len(data.get('inventoryItems', []))} inventory items from eBay")
                return data
            
            elif response.status_code == 204:
                # No content - empty inventory
                logger.info("No inventory items found in eBay account")
                return {"inventoryItems": [], "total": 0}
            
            elif response.status_code == 401:
                logger.error("eBay authentication failed - need user authorization")
                return {"inventoryItems": [], "total": 0, "error": "Authentication required"}
            
            else:
                logger.error(f"eBay API error: {response.status_code} - {response.text}")
                return {"inventoryItems": [], "total": 0, "error": f"API error: {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Error fetching eBay inventory items: {e}")
            return {"inventoryItems": [], "total": 0, "error": str(e)}
    
    def get_inventory_item(self, sku: str) -> Optional[Dict]:
        """
        Get specific inventory item by SKU
        
        Args:
            sku (str): SKU of the inventory item
            
        Returns:
            dict: Inventory item data or None if not found
        """
        
        if not self.is_available():
            return None
        
        try:
            response = self.oauth_service.make_authenticated_request(
                method='GET',
                endpoint=f'/inventory_item/{sku}',
                token_type='user'
            )
            
            if response and response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Inventory item {sku} not found")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching inventory item {sku}: {e}")
            return None
    
    def get_offers(self, limit: int = 50, offset: int = 0) -> Dict:
        """
        Get offers (eBay listings) from eBay
        
        Args:
            limit (int): Number of offers to retrieve
            offset (int): Offset for pagination
            
        Returns:
            dict: eBay API response with offers
        """
        
        if not self.is_available():
            return {"offers": [], "total": 0, "error": "Service not available"}
        
        params = {
            'limit': min(limit, 100),
            'offset': offset
        }
        
        try:
            response = self.oauth_service.make_authenticated_request(
                method='GET',
                endpoint='/offer',
                token_type='user',
                params=params
            )
            
            if response and response.status_code == 200:
                data = response.json()
                logger.info(f"Successfully retrieved {len(data.get('offers', []))} offers from eBay")
                return data
            elif response and response.status_code == 204:
                return {"offers": [], "total": 0}
            else:
                logger.error(f"Failed to fetch offers: {response.status_code if response else 'No response'}")
                return {"offers": [], "total": 0, "error": "Failed to fetch offers"}
                
        except Exception as e:
            logger.error(f"Error fetching eBay offers: {e}")
            return {"offers": [], "total": 0, "error": str(e)}
    
    def get_inventory_locations(self) -> Dict:
        """
        Get inventory locations from eBay
        
        Returns:
            dict: eBay API response with inventory locations
        """
        
        if not self.is_available():
            return {"locations": [], "error": "Service not available"}
        
        try:
            response = self.oauth_service.make_authenticated_request(
                method='GET',
                endpoint='/location',
                token_type='user'
            )
            
            if response and response.status_code == 200:
                data = response.json()
                locations = data.get('locations', [])
                logger.info(f"Successfully retrieved {len(locations)} inventory locations from eBay")
                return {"locations": locations}
            elif response and response.status_code == 204:
                return {"locations": []}
            else:
                logger.error(f"Failed to fetch locations: {response.status_code if response else 'No response'}")
                return {"locations": [], "error": "Failed to fetch locations"}
                
        except Exception as e:
            logger.error(f"Error fetching eBay inventory locations: {e}")
            return {"locations": [], "error": str(e)}
    
    def create_test_inventory_item(self, sku: str, product_data: Dict) -> Dict:
        """
        Create a test inventory item in eBay (for demo purposes)
        
        Args:
            sku (str): SKU for the item
            product_data (dict): Product information
            
        Returns:
            dict: Result of creation attempt
        """
        
        if not self.is_available():
            return {"success": False, "error": "Service not available"}
        
        # Build inventory item payload according to eBay API spec
        inventory_item = {
            "availability": {
                "shipToLocationAvailability": {
                    "quantity": product_data.get('quantity', 1)
                }
            },
            "condition": product_data.get('condition', 'NEW'),
            "product": {
                "title": product_data.get('title', f'Test Product {sku}'),
                "description": product_data.get('description', 'Test inventory item created via API'),
                "aspects": {
                    "Brand": [product_data.get('brand', 'Generic')],
                    "Type": [product_data.get('type', 'Test Item')]
                },
                "imageUrls": product_data.get('imageUrls', [])
            }
        }
        
        try:
            response = self.oauth_service.make_authenticated_request(
                method='PUT',
                endpoint=f'/inventory_item/{sku}',
                token_type='user',
                json=inventory_item
            )
            
            if response and response.status_code in [200, 201, 204]:
                logger.info(f"Successfully created/updated inventory item {sku}")
                return {"success": True, "sku": sku}
            else:
                error_msg = f"Failed to create inventory item: {response.status_code if response else 'No response'}"
                if response:
                    try:
                        error_data = response.json()
                        error_msg += f" - {error_data}"
                    except:
                        error_msg += f" - {response.text}"
                logger.error(error_msg)
                return {"success": False, "error": error_msg}
                
        except Exception as e:
            logger.error(f"Error creating inventory item {sku}: {e}")
            return {"success": False, "error": str(e)}
    
    def sync_inventory_from_ebay(self) -> Dict:
        """
        Sync inventory data from eBay to our local system
        
        Returns:
            dict: Sync results
        """
        
        if not self.is_available():
            return {
                "success": False,
                "error": "eBay service not available",
                "items_synced": 0,
                "offers_synced": 0
            }
        
        try:
            # Get inventory items
            items_response = self.get_inventory_items(limit=100)
            items = items_response.get('inventoryItems', [])
            
            # Get offers
            offers_response = self.get_offers(limit=100)
            offers = offers_response.get('offers', [])
            
            # Get locations
            locations_response = self.get_inventory_locations()
            locations = locations_response.get('locations', [])
            
            sync_results = {
                "success": True,
                "items_synced": len(items),
                "offers_synced": len(offers),
                "locations_synced": len(locations),
                "sync_timestamp": datetime.now().isoformat(),
                "details": {
                    "items": items[:5],  # Sample of first 5 items
                    "offers": offers[:5],  # Sample of first 5 offers
                    "locations": locations
                }
            }
            
            logger.info(f"eBay sync completed: {len(items)} items, {len(offers)} offers, {len(locations)} locations")
            return sync_results
            
        except Exception as e:
            logger.error(f"Error during eBay inventory sync: {e}")
            return {
                "success": False,
                "error": str(e),
                "items_synced": 0,
                "offers_synced": 0
            }
    
    def get_user_authorization_url(self) -> str:
        """
        Get eBay user authorization URL for OAuth flow
        
        Returns:
            str: Authorization URL for user to visit
        """
        return self.oauth_service.generate_user_authorization_url()
    
    def exchange_authorization_code(self, code: str) -> Dict:
        """
        Exchange authorization code for user token
        
        Args:
            code (str): Authorization code from eBay
            
        Returns:
            dict: Token exchange result
        """
        token_data = self.oauth_service.exchange_code_for_user_token(code)
        
        if token_data:
            return {
                "success": True,
                "message": "Successfully authorized with eBay",
                "expires_at": datetime.fromtimestamp(token_data['expires_at']).isoformat()
            }
        else:
            return {
                "success": False,
                "error": "Failed to exchange authorization code"
            }
    
    def get_service_status(self) -> Dict:
        """
        Get comprehensive status of eBay integration
        
        Returns:
            dict: Service status information
        """
        oauth_status = self.oauth_service.get_token_status()
        
        status = {
            "service_available": self.is_available(),
            "environment": self.environment,
            "oauth_status": oauth_status,
            "last_checked": datetime.now().isoformat()
        }
        
        # Test connectivity if credentials are available
        if oauth_status['credentials_configured']:
            try:
                test_response = self.get_inventory_items(limit=1)
                status["connectivity_test"] = {
                    "success": "error" not in test_response,
                    "message": test_response.get("error", "Connection successful")
                }
            except Exception as e:
                status["connectivity_test"] = {
                    "success": False,
                    "message": str(e)
                }
        
        return status


# Example usage and testing
if __name__ == "__main__":
    # Initialize service
    ebay_service = EbayInventoryService(environment="sandbox")
    
    # Check service status
    status = ebay_service.get_service_status()
    print("eBay Service Status:")
    print(json.dumps(status, indent=2))
    
    # Test inventory fetch if available
    if ebay_service.is_available():
        print("\nTesting inventory fetch...")
        items = ebay_service.get_inventory_items(limit=5)
        print(f"Retrieved {len(items.get('inventoryItems', []))} items")
        
        # Test sync
        print("\nTesting full sync...")
        sync_results = ebay_service.sync_inventory_from_ebay()
        print(f"Sync results: {sync_results['success']}")
    else:
        print("\neBay service not available - need to configure credentials")
        auth_url = ebay_service.get_user_authorization_url()
        if auth_url:
            print(f"Authorization URL: {auth_url}") 