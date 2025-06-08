#!/usr/bin/env python3
"""
eBay OAuth 2.0 Service
Handles eBay API authentication for Inventory Management System
Based on eBay's official OAuth implementation patterns
"""

import os
import base64
import time
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EbayOAuthService:
    """
    eBay OAuth 2.0 Service for managing Application and User tokens
    Supports both Sandbox and Production environments
    """
    
    def __init__(self, environment="sandbox"):
        """
        Initialize the eBay OAuth service
        
        Args:
            environment (str): 'sandbox' or 'production'
        """
        self.environment = environment.lower()
        
        # eBay API URLs
        if self.environment == "sandbox":
            self.token_url = "https://api.sandbox.ebay.com/identity/v1/oauth2/token"
            self.authorization_url = "https://auth.sandbox.ebay.com/oauth2/authorize"
            self.inventory_api_base = "https://api.sandbox.ebay.com/sell/inventory/v1"
        else:
            self.token_url = "https://api.ebay.com/identity/v1/oauth2/token"
            self.authorization_url = "https://auth.ebay.com/oauth2/authorize"
            self.inventory_api_base = "https://api.ebay.com/sell/inventory/v1"
        
        # Load credentials
        self.client_id = None
        self.client_secret = None
        self.redirect_uri = None
        self.load_credentials()
        
        # Token storage paths
        self.token_dir = os.path.join(os.path.dirname(__file__), 'tokens')
        os.makedirs(self.token_dir, exist_ok=True)
        
        self.app_token_file = os.path.join(self.token_dir, f'ebay_app_token_{environment}.json')
        self.user_token_file = os.path.join(self.token_dir, f'ebay_user_token_{environment}.json')
    
    def load_credentials(self):
        """Load eBay API credentials from environment variables or config files"""
        
        # Try environment variables first
        self.client_id = os.getenv('EBAY_CLIENT_ID') or os.getenv('EBAY_APP_ID')
        self.client_secret = os.getenv('EBAY_CLIENT_SECRET') or os.getenv('EBAY_CERT_ID')
        # For eBay OAuth, redirect_uri should be the RuName, not the URL
        self.redirect_uri = os.getenv('EBAY_RUNAME') or os.getenv('EBAY_REDIRECT_URI')
        
        # If not found, try config files
        if not self.client_id:
            config_dir = os.path.join(os.path.dirname(__file__), 'config')
            try:
                with open(os.path.join(config_dir, 'ebay_client_id.txt'), 'r') as f:
                    self.client_id = f.read().strip()
                with open(os.path.join(config_dir, 'ebay_client_secret.txt'), 'r') as f:
                    self.client_secret = f.read().strip()
                with open(os.path.join(config_dir, 'ebay_redirect_uri.txt'), 'r') as f:
                    self.redirect_uri = f.read().strip()
            except FileNotFoundError:
                logger.warning("eBay credentials not found in environment or config files")
        
        if self.client_id:
            logger.info(f"Loaded eBay credentials for {self.environment} environment")
        else:
            logger.warning("eBay credentials not configured - using demo mode")
    
    def get_application_token(self, force_refresh=False) -> Optional[str]:
        """
        Get or refresh Application access token (Client Credentials Grant)
        Used for public data access (browsing, searching)
        
        Args:
            force_refresh (bool): Force token refresh even if current token is valid
            
        Returns:
            str: Access token or None if failed
        """
        
        # Check if we have credentials
        if not self.client_id or not self.client_secret:
            logger.warning("No eBay credentials available for Application token")
            return None
        
        # Check existing token
        if not force_refresh:
            existing_token = self._load_token(self.app_token_file)
            if existing_token and self._is_token_valid(existing_token):
                logger.info("Using existing Application token")
                return existing_token['access_token']
        
        # Request new token
        logger.info("Requesting new Application token from eBay")
        
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Authorization': 'Basic ' + base64.b64encode(
                f"{self.client_id}:{self.client_secret}".encode()
            ).decode()
        }
        
        data = {
            'grant_type': 'client_credentials',
            'scope': 'https://api.ebay.com/oauth/api_scope'
        }
        
        try:
            response = requests.post(self.token_url, headers=headers, data=data, timeout=30)
            response.raise_for_status()
            
            token_data = response.json()
            
            # Add expiration timestamp
            token_data['expires_at'] = time.time() + token_data.get('expires_in', 7200)
            token_data['token_type'] = 'application'
            
            # Save token
            self._save_token(self.app_token_file, token_data)
            
            logger.info("Successfully obtained Application token")
            return token_data['access_token']
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get Application token: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting Application token: {e}")
            return None
    
    def generate_user_authorization_url(self, scopes: list = None) -> str:
        """
        Generate URL for user authorization (first step of Authorization Code Grant)
        
        Args:
            scopes (list): List of OAuth scopes needed
            
        Returns:
            str: Authorization URL for user to visit
        """
        
        if not self.client_id or not self.redirect_uri:
            logger.error("Missing client_id or redirect_uri for user authorization")
            return ""
        
        # Default scopes for inventory management
        if scopes is None:
            scopes = [
                'https://api.ebay.com/oauth/api_scope',
                'https://api.ebay.com/oauth/api_scope/sell.inventory',
                'https://api.ebay.com/oauth/api_scope/sell.inventory.readonly',
                'https://api.ebay.com/oauth/api_scope/sell.account',
                'https://api.ebay.com/oauth/api_scope/sell.account.readonly'
            ]
        
        # Import urllib.parse for proper URL encoding
        import urllib.parse
        
        # URL encode the components
        scope_string = ' '.join(scopes)
        encoded_redirect_uri = urllib.parse.quote(self.redirect_uri, safe='')
        encoded_scope = urllib.parse.quote(scope_string, safe='')
        
        # Generate authorization URL with proper encoding
        auth_url = (
            f"{self.authorization_url}"
            f"?client_id={self.client_id}"
            f"&response_type=code"
            f"&redirect_uri={encoded_redirect_uri}"
            f"&scope={encoded_scope}"
            f"&prompt=login"  # Force user to log in each time
        )
        
        logger.info(f"Generated user authorization URL for {len(scopes)} scopes")
        return auth_url
    
    def exchange_code_for_user_token(self, authorization_code: str, scopes: list = None) -> Optional[Dict]:
        """
        Exchange authorization code for User access token (Authorization Code Grant)
        
        Args:
            authorization_code (str): Code received from eBay after user authorization
            scopes (list): Same scopes used in authorization URL
            
        Returns:
            dict: Token data including access_token and refresh_token
        """
        
        if not self.client_id or not self.client_secret or not self.redirect_uri:
            logger.error("Missing credentials for User token exchange")
            return None
        
        # Default scopes
        if scopes is None:
            scopes = [
                'https://api.ebay.com/oauth/api_scope',
                'https://api.ebay.com/oauth/api_scope/sell.inventory',
                'https://api.ebay.com/oauth/api_scope/sell.inventory.readonly',
                'https://api.ebay.com/oauth/api_scope/sell.account',
                'https://api.ebay.com/oauth/api_scope/sell.account.readonly'
            ]
        
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Authorization': 'Basic ' + base64.b64encode(
                f"{self.client_id}:{self.client_secret}".encode()
            ).decode()
        }
        
        data = {
            'grant_type': 'authorization_code',
            'code': authorization_code,
            'redirect_uri': self.redirect_uri
        }
        
        try:
            response = requests.post(self.token_url, headers=headers, data=data, timeout=30)
            response.raise_for_status()
            
            token_data = response.json()
            
            # Add metadata
            token_data['expires_at'] = time.time() + token_data.get('expires_in', 7200)
            token_data['refresh_expires_at'] = time.time() + token_data.get('refresh_token_expires_in', 47304000)
            token_data['token_type'] = 'user'
            token_data['scopes'] = scopes
            
            # Save token
            self._save_token(self.user_token_file, token_data)
            
            logger.info("Successfully exchanged authorization code for User token")
            return token_data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to exchange code for User token: {e}")
            if hasattr(e, 'response') and e.response:
                logger.error(f"Response: {e.response.text}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error exchanging code: {e}")
            return None
    
    def get_user_token(self, force_refresh=False) -> Optional[str]:
        """
        Get valid User access token, refreshing if necessary
        
        Args:
            force_refresh (bool): Force token refresh
            
        Returns:
            str: Access token or None if failed
        """
        
        # Load existing token
        token_data = self._load_token(self.user_token_file)
        
        if not token_data:
            logger.warning("No User token found - need user authorization first")
            return None
        
        # Check if token is still valid
        if not force_refresh and self._is_token_valid(token_data):
            logger.info("Using existing User token")
            return token_data['access_token']
        
        # Try to refresh the token
        if 'refresh_token' in token_data:
            logger.info("Refreshing User token")
            refreshed_token = self._refresh_user_token(token_data)
            if refreshed_token:
                return refreshed_token['access_token']
        
        logger.warning("User token expired and cannot be refreshed - need new authorization")
        return None
    
    def _refresh_user_token(self, token_data: Dict) -> Optional[Dict]:
        """
        Refresh User access token using refresh token
        
        Args:
            token_data (dict): Current token data with refresh_token
            
        Returns:
            dict: New token data or None if failed
        """
        
        if not self.client_id or not self.client_secret:
            logger.error("Missing credentials for token refresh")
            return None
        
        refresh_token = token_data.get('refresh_token')
        scopes = token_data.get('scopes', ['https://api.ebay.com/oauth/api_scope'])
        
        if not refresh_token:
            logger.error("No refresh token available")
            return None
        
        # Check if refresh token is still valid
        refresh_expires_at = token_data.get('refresh_expires_at', 0)
        if time.time() > refresh_expires_at:
            logger.error("Refresh token has expired")
            return None
        
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Authorization': 'Basic ' + base64.b64encode(
                f"{self.client_id}:{self.client_secret}".encode()
            ).decode()
        }
        
        data = {
            'grant_type': 'refresh_token',
            'refresh_token': refresh_token,
            'scope': ' '.join(scopes)
        }
        
        try:
            response = requests.post(self.token_url, headers=headers, data=data, timeout=30)
            response.raise_for_status()
            
            new_token_data = response.json()
            
            # Add metadata
            new_token_data['expires_at'] = time.time() + new_token_data.get('expires_in', 7200)
            new_token_data['refresh_expires_at'] = time.time() + new_token_data.get('refresh_token_expires_in', 47304000)
            new_token_data['token_type'] = 'user'
            new_token_data['scopes'] = scopes
            
            # Save refreshed token
            self._save_token(self.user_token_file, new_token_data)
            
            logger.info("Successfully refreshed User token")
            return new_token_data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to refresh User token: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error refreshing token: {e}")
            return None
    
    def make_authenticated_request(self, method: str, endpoint: str, token_type: str = "user", **kwargs) -> Optional[requests.Response]:
        """
        Make authenticated request to eBay API
        
        Args:
            method (str): HTTP method (GET, POST, PUT, DELETE)
            endpoint (str): API endpoint path
            token_type (str): 'user' or 'application'
            **kwargs: Additional arguments for requests
            
        Returns:
            requests.Response: API response or None if failed
        """
        
        # Get appropriate token
        if token_type == "user":
            token = self.get_user_token()
        else:
            token = self.get_application_token()
        
        if not token:
            logger.error(f"No valid {token_type} token available")
            return None
        
        # Build full URL
        url = f"{self.inventory_api_base}{endpoint}"
        
        # Set headers
        headers = kwargs.get('headers', {})
        headers.update({
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        kwargs['headers'] = headers
        
        try:
            response = requests.request(method, url, timeout=30, **kwargs)
            
            # Check for token expiration
            if response.status_code == 401:
                logger.info(f"{token_type.title()} token expired, attempting to refresh")
                
                # Try to refresh and retry
                if token_type == "user":
                    token = self.get_user_token(force_refresh=True)
                else:
                    token = self.get_application_token(force_refresh=True)
                
                if token:
                    headers['Authorization'] = f'Bearer {token}'
                    response = requests.request(method, url, timeout=30, **kwargs)
            
            return response
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None
    
    def _load_token(self, token_file: str) -> Optional[Dict]:
        """Load token from file"""
        try:
            if os.path.exists(token_file):
                with open(token_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading token from {token_file}: {e}")
        return None
    
    def _save_token(self, token_file: str, token_data: Dict):
        """Save token to file"""
        try:
            with open(token_file, 'w') as f:
                json.dump(token_data, f, indent=2)
            logger.info(f"Token saved to {token_file}")
        except Exception as e:
            logger.error(f"Error saving token to {token_file}: {e}")
    
    def _is_token_valid(self, token_data: Dict, buffer_seconds: int = 300) -> bool:
        """
        Check if token is still valid
        
        Args:
            token_data (dict): Token data with expires_at timestamp
            buffer_seconds (int): Buffer time before expiration
            
        Returns:
            bool: True if token is valid
        """
        expires_at = token_data.get('expires_at', 0)
        return time.time() < (expires_at - buffer_seconds)
    
    def get_token_status(self) -> Dict:
        """
        Get status of all tokens
        
        Returns:
            dict: Status information for all tokens
        """
        status = {
            'credentials_configured': bool(self.client_id and self.client_secret),
            'environment': self.environment,
            'application_token': {'status': 'not_found'},
            'user_token': {'status': 'not_found'}
        }
        
        # Check Application token
        app_token = self._load_token(self.app_token_file)
        if app_token:
            if self._is_token_valid(app_token):
                status['application_token'] = {
                    'status': 'valid',
                    'expires_at': datetime.fromtimestamp(app_token['expires_at']).isoformat()
                }
            else:
                status['application_token'] = {
                    'status': 'expired',
                    'expires_at': datetime.fromtimestamp(app_token['expires_at']).isoformat()
                }
        
        # Check User token
        user_token = self._load_token(self.user_token_file)
        if user_token:
            if self._is_token_valid(user_token):
                status['user_token'] = {
                    'status': 'valid',
                    'expires_at': datetime.fromtimestamp(user_token['expires_at']).isoformat(),
                    'scopes': user_token.get('scopes', [])
                }
            else:
                refresh_expires_at = user_token.get('refresh_expires_at', 0)
                if time.time() < refresh_expires_at:
                    status['user_token'] = {
                        'status': 'expired_but_refreshable',
                        'expires_at': datetime.fromtimestamp(user_token['expires_at']).isoformat(),
                        'refresh_expires_at': datetime.fromtimestamp(refresh_expires_at).isoformat()
                    }
                else:
                    status['user_token'] = {
                        'status': 'expired',
                        'expires_at': datetime.fromtimestamp(user_token['expires_at']).isoformat()
                    }
        
        return status

    def get_seller_info(self, access_token: str) -> Dict:
        """
        Get seller information using the access token
        
        Args:
            access_token (str): Valid eBay access token
            
        Returns:
            dict: Seller information
        """
        try:
            # Use eBay Commerce Account API to get seller info
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
            
            # Get user information from eBay
            if self.environment == "sandbox":
                user_url = "https://apiz.sandbox.ebay.com/commerce/identity/v1/user/"
            else:
                user_url = "https://apiz.ebay.com/commerce/identity/v1/user/"
            
            response = requests.get(user_url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                user_data = response.json()
                logger.info("Successfully retrieved seller information")
                return {
                    'sellerId': user_data.get('userId', 'unknown'),
                    'username': user_data.get('username', 'unknown'),
                    'email': user_data.get('email', ''),
                    'registrationMarketplaceId': user_data.get('registrationMarketplaceId', 'EBAY_US')
                }
            else:
                logger.warning(f"Failed to get seller info: {response.status_code}")
                # Return default seller info if API call fails
                return {
                    'sellerId': 'demo_seller_' + str(int(time.time())),
                    'username': 'demo_user',
                    'email': '',
                    'registrationMarketplaceId': 'EBAY_US'
                }
                
        except Exception as e:
            logger.error(f"Error getting seller info: {e}")
            # Return default seller info on error
            return {
                'sellerId': 'demo_seller_' + str(int(time.time())),
                'username': 'demo_user', 
                'email': '',
                'registrationMarketplaceId': 'EBAY_US'
            }

    def store_tokens(self, token_data: Dict, seller_info: Dict) -> bool:
        """
        Store tokens and seller information in the database
        
        Args:
            token_data (dict): Token data from eBay OAuth
            seller_info (dict): Seller information
            
        Returns:
            bool: True if successful
        """
        try:
            from models import get_db_session, EBayToken, EBaySeller
            from datetime import datetime, timedelta
            
            session = get_db_session()
            
            # Create or update seller record
            seller = session.query(EBaySeller).filter_by(seller_id=seller_info['sellerId']).first()
            if not seller:
                seller = EBaySeller(
                    seller_id=seller_info['sellerId'],
                    seller_username=seller_info.get('username', ''),
                    marketplace_id=seller_info.get('registrationMarketplaceId', 'EBAY_US'),
                    created_at=datetime.utcnow()
                )
                session.add(seller)
            else:
                # Update existing seller
                seller.seller_username = seller_info.get('username', seller.seller_username)
                seller.marketplace_id = seller_info.get('registrationMarketplaceId', seller.marketplace_id)
                seller.updated_at = datetime.utcnow()
            
            # Deactivate old tokens for this seller
            session.query(EBayToken).filter_by(seller_id=seller_info['sellerId']).update({'is_active': False})
            
            # Create new token record
            expires_at = datetime.utcnow() + timedelta(seconds=token_data.get('expires_in', 7200))
            refresh_expires_at = datetime.utcnow() + timedelta(seconds=token_data.get('refresh_token_expires_in', 47304000))
            
            token = EBayToken(
                seller_id=seller_info['sellerId'],
                access_token=token_data['access_token'],
                refresh_token=token_data.get('refresh_token') or None,  # Convert empty string to None
                token_expires_at=expires_at,
                refresh_expires_at=refresh_expires_at,
                scope=token_data.get('scope') or None,  # Convert empty string to None
                token_type='user',
                is_active=True,
                created_at=datetime.utcnow()
            )
            session.add(token)
            
            # Save tokens to file as well (for backup)
            self._save_token(self.user_token_file, token_data)
            
            session.commit()
            session.close()
            
            logger.info(f"Successfully stored tokens for seller: {seller_info['sellerId']}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing tokens: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Token data keys: {list(token_data.keys()) if token_data else 'None'}")
            logger.error(f"Seller info: {seller_info}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            try:
                session.rollback()
                session.close()
            except:
                pass
            return False

    def test_token_validity(self, access_token: str) -> bool:
        """
        Test if an access token is still valid by making a simple API call
        
        Args:
            access_token (str): Access token to test
            
        Returns:
            bool: True if token is valid
        """
        try:
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
            
            # Make a simple API call to test the token
            if self.environment == "sandbox":
                test_url = "https://api.sandbox.ebay.com/sell/inventory/v1/inventory_item"
            else:
                test_url = "https://api.ebay.com/sell/inventory/v1/inventory_item"
            
            response = requests.get(test_url, headers=headers, timeout=10)
            
            # Token is valid if we don't get a 401 Unauthorized
            return response.status_code != 401
            
        except Exception as e:
            logger.error(f"Error testing token validity: {e}")
            return False


# Example usage and testing
if __name__ == "__main__":
    # Initialize OAuth service
    oauth = EbayOAuthService(environment="sandbox")
    
    # Get token status
    status = oauth.get_token_status()
    print("Token Status:")
    print(json.dumps(status, indent=2))
    
    # Try to get Application token
    app_token = oauth.get_application_token()
    if app_token:
        print(f"Application token obtained: {app_token[:20]}...")
    else:
        print("Failed to get Application token")
    
    # Generate user authorization URL
    auth_url = oauth.generate_user_authorization_url()
    if auth_url:
        print(f"User authorization URL: {auth_url}")
    else:
        print("Failed to generate authorization URL") 