#!/usr/bin/env python3
"""
Industry-based Discord channel routing system
Routes alerts to different Discord channels based on industry classifications
"""

import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

from data_access.document_store import load_document
from data_access.metadata_repository import fetch_stock_metadata_map
from discord_rate_limiter import get_rate_limiter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DiscordEconomyRouter:
    """
    Routes Discord alerts to different channels based on economy classifications
    """
    
    def __init__(self, config_file: str = 'discord_channels_config.json', use_rate_limiter: bool = True):
        """
        Initialize the Discord router with channel configuration

        Args:
            config_file: Path to the Discord channels configuration file
            use_rate_limiter: Whether to use rate limiting for Discord webhooks
        """
        self.config_file = config_file
        self.config = self._load_config()
        self.stock_data = self._load_stock_data()
        self.custom_channels_file = 'custom_discord_channels.json'
        self.custom_channels = self._load_custom_channels()
        self.use_rate_limiter = use_rate_limiter
        self.rate_limiter = get_rate_limiter() if use_rate_limiter else None

    def _load_config(self) -> Dict:
        """Load Discord channel configuration"""
        try:
            config = load_document(
                "discord_channels_config",
                default={},
                fallback_path=self.config_file,
            ) or {}
            logger.info(f"Loaded Discord routing configuration from {self.config_file}")

            # Ensure hourly mappings exist and inherit missing fields/webhooks from the daily map
            daily_mappings = config.get('channel_mappings', {}) or {}
            hourly_mappings = config.get('channel_mappings_hourly') or {}
            for key, value in daily_mappings.items():
                hourly_entry = hourly_mappings.get(key, {})

                # Keep existing hourly channel name if provided; otherwise derive from daily
                channel_name = hourly_entry.get('channel_name')
                if not channel_name:
                    base_channel = value.get('channel_name', '')
                    if base_channel and base_channel.endswith('-alerts'):
                        channel_name = base_channel[:-7] + '-hourly-alerts'
                    elif base_channel.startswith('#'):
                        channel_name = f"{base_channel}-hourly" if base_channel else ''
                    else:
                        channel_name = f"{base_channel}-hourly" if base_channel else ''

                description = hourly_entry.get('description') or value.get('description', '')
                if description and 'Hourly' not in description:
                    description = f"{description} (Hourly)"
                elif not description:
                    description = "Hourly alerts"

                # Inherit the daily webhook when the hourly one is blank/missing to avoid silent drops
                webhook_url = hourly_entry.get('webhook_url') or value.get('webhook_url', '')

                hourly_mappings[key] = {
                    "webhook_url": webhook_url,
                    "channel_name": channel_name,
                    "description": description,
                }

            config['channel_mappings_hourly'] = hourly_mappings

            return config
        except FileNotFoundError:
            logger.warning(f"Discord config file {self.config_file} not found. Using default configuration.")
            return self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading Discord routing config: {e}")
            return self._get_default_config()

    def _reload_configs_if_changed(self) -> None:
        """Refresh channel configurations from persistent storage."""
        self.config = self._load_config()
        self.custom_channels = self._load_custom_channels()

    def _get_default_config(self) -> Dict:
        """Get default configuration when config file is missing"""
        return {
            "channel_mappings": {
                "General": {
                    "webhook_url": "YOUR_GENERAL_WEBHOOK_URL_HERE",
                    "channel_name": "#general-alerts",
                    "description": "General alerts and fallback"
                }
            },
            "default_channel": "General",
            "enable_industry_routing": False,
            "log_routing_decisions": True
        }
    
    def _load_stock_data(self) -> pd.DataFrame:
        """Load stock data with industry classifications"""
        metadata = fetch_stock_metadata_map() or {}
        if metadata:
            records = []
            for symbol, info in metadata.items():
                records.append(
                    {
                        'Symbol': symbol,
                        'Name': info.get('name', ''),
                        'Economy': info.get('rbics_economy', ''),
                        'AssetType': info.get('asset_type', 'Stock'),
                    }
                )
            df = pd.DataFrame(records)
            logger.info("Loaded %s stocks with industry data from PostgreSQL", len(df))
            return df

        try:
            df = pd.read_csv('cleaned_data.csv')
            logger.info(f"Loaded {len(df)} stocks with industry data from cleaned_data.csv")
            return df
        except FileNotFoundError:
            logger.warning("Neither PostgreSQL metadata nor cleaned_data.csv found. Industry routing will be limited.")
            return pd.DataFrame()
    
    def _load_custom_channels(self) -> Dict:
        """Load custom Discord channels configuration"""
        try:
            channels = load_document(
                "custom_discord_channels",
                default={},
                fallback_path=self.custom_channels_file,
            ) or {}
            logger.info(f"Loaded {len(channels)} custom Discord channels")
            return channels
        except Exception as e:
            logger.error(f"Error loading custom channels configuration: {e}")
            return {}
    
    def check_custom_channel_condition(self, alert: Dict, channel_info: Dict) -> bool:
        """
        Check if an alert's condition matches a custom channel's condition

        Args:
            alert: Alert dictionary containing a 'condition' field or 'conditions' array
            channel_info: Custom channel configuration with a 'condition' field

        Returns:
            True if alert condition matches channel condition, False otherwise
        """
        if not channel_info.get('enabled', False):
            return False

        channel_condition = channel_info.get('condition', '')

        # If channel still has old format (condition_type/condition_value), skip it
        if 'condition_type' in channel_info and 'condition' not in channel_info:
            return False

        if not channel_condition:
            return False

        # Normalize conditions for comparison
        # Remove all spaces around operators and make lowercase
        def normalize_condition(cond):
            # Remove spaces around operators and parentheses
            import re
            cond = re.sub(r'\s*([<>=!]+)\s*', r'\1', cond)
            cond = re.sub(r'\s*([(),\[\]])\s*', r'\1', cond)
            cond = re.sub(r'\s+', ' ', cond)  # Collapse multiple spaces
            return cond.strip().lower()

        def is_price_level_condition(cond):
            """Check if a condition is a price level comparison"""
            import re
            # Pattern matches: Close/Open/High/Low followed by comparison operator and a number
            # Examples: "Close[-1] < 26", "Close > 100.50", "Open[-1] >= 50"
            price_pattern = r'(close|open|high|low)(\[-?\d+\])?\s*([<>=!]+)\s*\$?\d+\.?\d*'
            return bool(re.search(price_pattern, cond.lower()))

        channel_condition_normalized = normalize_condition(channel_condition)

        # Check if channel condition is "price_level" - special keyword for ANY price level condition
        if channel_condition_normalized == 'price_level':
            # Get alert conditions
            alert_conditions = []

            # Get single condition if exists
            if alert.get('condition'):
                alert_conditions.append(alert.get('condition'))

            # Get conditions array if exists
            if 'conditions' in alert:
                conditions_list = alert.get('conditions', [])
                for cond_item in conditions_list:
                    if isinstance(cond_item, dict) and 'conditions' in cond_item:
                        alert_conditions.append(cond_item['conditions'])
                    elif isinstance(cond_item, list) and len(cond_item) > 0:
                        alert_conditions.append(cond_item[0] if isinstance(cond_item[0], str) else str(cond_item[0]))
                    elif isinstance(cond_item, str):
                        alert_conditions.append(cond_item)

            # Check if ANY alert condition is a price level condition
            for alert_cond in alert_conditions:
                if alert_cond and is_price_level_condition(alert_cond):
                    return True

            return False

        # Otherwise, do exact condition matching
        # Get the condition strings from alert
        # Handle both 'condition' (singular) and 'conditions' (array) format
        alert_condition = alert.get('condition', '')

        # Check if single condition field matches
        if alert_condition:
            alert_condition_normalized = normalize_condition(alert_condition)
            if alert_condition_normalized == channel_condition_normalized:
                return True

        # If no 'condition' field, try to check conditions array
        if 'conditions' in alert:
            conditions_list = alert.get('conditions', [])
            for cond_item in conditions_list:
                # Handle dict format
                if isinstance(cond_item, dict) and 'conditions' in cond_item:
                    alert_cond = cond_item['conditions']
                # Handle nested list format (e.g., [['condition string']])
                elif isinstance(cond_item, list) and len(cond_item) > 0:
                    alert_cond = cond_item[0] if isinstance(cond_item[0], str) else str(cond_item[0])
                # Handle simple string in list
                elif isinstance(cond_item, str):
                    alert_cond = cond_item
                else:
                    continue

                if alert_cond:
                    alert_cond_normalized = normalize_condition(alert_cond)
                    if alert_cond_normalized == channel_condition_normalized:
                        return True

        return False
    
    def get_custom_channels_for_alert(self, alert: Dict) -> List[Tuple[str, Dict]]:
        """
        Get all custom channels that match the alert's conditions
        
        Args:
            alert: Alert dictionary
            
        Returns:
            List of (channel_name, channel_info) tuples for matching channels
        """
        self._reload_configs_if_changed()
        matching_channels = []
        
        for channel_name, channel_info in self.custom_channels.items():
            if self.check_custom_channel_condition(alert, channel_info):
                matching_channels.append((channel_name, channel_info))
                if self.config.get('log_routing_decisions', False):
                    condition = channel_info.get('condition', 'Unknown')
                    # Get alert condition for logging
                    alert_cond_str = alert.get('condition', '')
                    if not alert_cond_str and 'conditions' in alert:
                        conditions_list = alert.get('conditions', [])
                        if conditions_list and len(conditions_list) > 0:
                            for cond_item in conditions_list:
                                if isinstance(cond_item, dict) and 'conditions' in cond_item:
                                    alert_cond_str = cond_item['conditions']
                                    break
                    logger.info(f"Alert condition '{alert_cond_str}' matched custom channel '{channel_name}' (condition: {condition})")
        
        return matching_channels
    
    def get_stock_economy(self, ticker: str) -> Optional[str]:
        """
        Get the economy classification for a given stock ticker
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Economy classification or None if not found
        """
        if self.stock_data.empty:
            return None
            
        # Clean ticker for matching
        clean_ticker = ticker.split('.')[0] if '.' in ticker else ticker
        
        # Try exact match first
        match = self.stock_data[self.stock_data['Symbol'] == ticker]
        if not match.empty and 'Economy' in match.columns:
            economy = match.iloc[0]['Economy']
            if pd.notna(economy) and economy != '':
                return economy
        
        # Try with uppercase
        match = self.stock_data[self.stock_data['Symbol'] == ticker.upper()]
        if not match.empty and 'Economy' in match.columns:
            economy = match.iloc[0]['Economy']
            if pd.notna(economy) and economy != '':
                return economy
        
        # Try partial match
        match = self.stock_data[self.stock_data['Symbol'].str.contains(clean_ticker, case=False, na=False)]
        if not match.empty and 'Economy' in match.columns:
            economy = match.iloc[0]['Economy']
            if pd.notna(economy) and economy != '':
                return economy
        
        return None
    
    def is_etf(self, ticker: str) -> bool:
        """
        Determine if a ticker is an ETF based on asset_type field
        
        Args:
            ticker: Stock ticker
            
        Returns:
            True if ETF, False otherwise
        """
        if self.stock_data.empty:
            return False
            
        # Try exact match
        match = self.stock_data[self.stock_data['Symbol'] == ticker]
        if not match.empty and 'AssetType' in match.columns:
            return match.iloc[0]['AssetType'] == 'ETF'
        
        # Try uppercase match
        match = self.stock_data[self.stock_data['Symbol'] == ticker.upper()]
        if not match.empty and 'AssetType' in match.columns:
            return match.iloc[0]['AssetType'] == 'ETF'
        
        # Try with -US suffix if not present
        if '-' not in ticker:
            ticker_with_suffix = f"{ticker}-US"
            match = self.stock_data[self.stock_data['Symbol'] == ticker_with_suffix]
            if not match.empty and 'AssetType' in match.columns:
                return match.iloc[0]['AssetType'] == 'ETF'
        
        return False

    def is_futures(self, ticker: str) -> bool:
        """
        Determine if a ticker is a futures contract

        Args:
            ticker: Symbol to check

        Returns:
            True if futures, False otherwise
        """
        # Import futures backend
        try:
            from backend_futures_flexible import is_futures_symbol
            return is_futures_symbol(ticker)
        except ImportError:
            # Fallback to basic checks
            ticker_upper = ticker.upper()
            # Check for common futures indicators
            if ticker_upper.endswith('=F'):
                return True
            # Check for known futures symbols
            futures_symbols = ['CL', 'GC', 'SI', 'ES', 'NQ', 'YM', 'ZC', 'ZW', 'ZS', 'NG', 'HG', 'DX']
            if ticker_upper in futures_symbols:
                return True
            return False


    def _normalize_timeframe(self, timeframe: Optional[str]) -> str:
        """Normalize timeframe strings into daily/weekly/hourly buckets."""
        if not timeframe:
            return 'daily'
        tf = str(timeframe).lower()
        if tf in {'1wk', '1w', 'weekly', 'week'}:
            return 'weekly'
        if tf in {'1h', '1hr', 'hourly', 'hour'}:
            return 'hourly'
        return 'daily'

    def _get_channel_mapping(self, timeframe: Optional[str]) -> Tuple[Dict, str]:
        """Return the channel mapping dictionary and config key for a timeframe."""
        normalized = self._normalize_timeframe(timeframe)
        if normalized == 'hourly':
            mapping = self.config.get('channel_mappings_hourly')
            if mapping:
                return mapping, 'channel_mappings_hourly'
        return self.config.get('channel_mappings', {}), 'channel_mappings'

    def determine_alert_channel(self, alert: Dict) -> Tuple[str, str]:
        """
        Determine which Discord channel to send an alert to based on economy classification
        
        Args:
            alert: Alert dictionary containing stock information
            
        Returns:
            Tuple of (channel_name, webhook_url)
        """
        if not self.config.get('enable_industry_routing', False):
            default_channel = self.config.get('default_channel', 'General')
            return self._get_channel_info(default_channel)
        
        # Handle ratio alerts - ALL ratio alerts go to Pairs channel
        alert_timeframe = alert.get('timeframe', '1d')
        mapping, _ = self._get_channel_mapping(alert_timeframe)

        if alert.get('ratio') == 'Yes':
            ticker1 = alert.get('ticker1', '')
            ticker2 = alert.get('ticker2', '')
            
            if self.config.get('log_routing_decisions', False):
                logger.info(f"Ratio alert {ticker1}/{ticker2} routed to Pairs channel")
            return self._get_channel_info('Pairs', alert_timeframe)
        
        # Handle single stock alerts
        ticker = alert.get('ticker', '')
        name = alert.get('stock_name', alert.get('name', ''))
        
        if not ticker:
            return self._get_channel_info(self.config.get('default_channel', 'General'))
        
        # Add -US suffix if not present for US stocks
        if '-' not in ticker and not ticker.endswith('.'):
            ticker_with_suffix = f"{ticker}-US"
        else:
            ticker_with_suffix = ticker
        
        # Futures routing has been disabled - all assets will be routed by economy/industry
        # Previously futures were routed to a separate channel, but this has been removed

        # Check if it's an ETF using asset_type field
        if self.is_etf(ticker):
            if self.config.get('log_routing_decisions', False):
                logger.info(f"ETF {ticker} routed to ETFs channel")
            return self._get_channel_info('ETFs', alert_timeframe)

        # Get economy classification (try both with and without suffix)
        economy = self.get_stock_economy(ticker)
        if not economy:
            economy = self.get_stock_economy(ticker_with_suffix)
        
        if economy and economy in mapping:
            if self.config.get('log_routing_decisions', False):
                logger.info(f"Stock {ticker} ({economy}) routed to {economy} channel")
            return self._get_channel_info(economy, alert_timeframe)
        
        # Fallback to default channel
        if self.config.get('log_routing_decisions', False):
            logger.info(f"Stock {ticker} routed to default channel (economy: {economy})")
        return self._get_channel_info(self.config.get('default_channel', 'General'), alert_timeframe)
    
    def _get_channel_info(self, channel_name: str, timeframe: Optional[str] = None) -> Tuple[str, str]:
        """
        Get channel information from configuration
        
        Args:
            channel_name: Name of the channel
            
        Returns:
            Tuple of (channel_name, webhook_url)
        """
        mapping, mapping_key = self._get_channel_mapping(timeframe)

        if channel_name in mapping:
            channel_info = mapping[channel_name]
            return channel_info['channel_name'], channel_info['webhook_url']
        
        # Fallback to default
        default_channel = self.config.get('default_channel', 'General')
        fallback_mapping = self.config.get('channel_mappings', {})
        if default_channel in fallback_mapping:
            channel_info = fallback_mapping[default_channel]
            return channel_info['channel_name'], channel_info['webhook_url']
        
        # Ultimate fallback
        return "#general-alerts", "YOUR_GENERAL_WEBHOOK_URL_HERE"
    
    def send_economy_alert(self, alert: Dict, message: str) -> bool:
        """
        Send an alert to the appropriate Discord channel based on economy classification
        Also sends to any matching custom channels
        
        Args:
            alert: Alert dictionary
            message: Alert message to send
            
        Returns:
            True if sent successfully, False otherwise
        """
        self._reload_configs_if_changed()
        success = False
        channels_sent = []  # Track which channels received the alert

        # First, send to economy-based channel
        logger.info(f"Starting dual-channel send for alert: {alert.get('ticker', 'Unknown')}")
        try:
            alert_timeframe = alert.get('timeframe', '1d')
            channel_name, webhook_url = self.determine_alert_channel(alert)

            if not webhook_url:
                fallback_channel = self.config.get('default_channel', 'General')
                logger.warning(
                    "Discord webhook missing for %s (timeframe=%s); falling back to %s",
                    channel_name,
                    alert_timeframe,
                    fallback_channel,
                )
                channel_name, webhook_url = self._get_channel_info(fallback_channel, alert_timeframe)

            if not webhook_url:
                logger.error(
                    "No Discord webhook available for %s (timeframe=%s); skipping alert send",
                    channel_name,
                    alert_timeframe,
                )
                return False
            
            # Check if webhook URL is configured
            if webhook_url == "YOUR_GENERAL_WEBHOOK_URL_HERE" or "YOUR_" in webhook_url:
                logger.warning(f"Discord webhook not configured for {channel_name}. Please update {self.config_file}")
            else:
                # Send to Discord with optional spacing
                spacing_config = self.config.get('message_spacing', {})
                if spacing_config.get('enabled', False):
                    divider_char = spacing_config.get('divider_char', '━')
                    divider_length = spacing_config.get('divider_length', 30)
                    divider = divider_char * divider_length
                    formatted_message = f"{divider}\n**{channel_name}**\n{message}\n{divider}"
                else:
                    formatted_message = f"**{channel_name}**\n{message}"
                
                payload = {
                    "content": formatted_message,
                    "username": "Stock Alert Bot",
                    "avatar_url": "https://cdn.discordapp.com/attachments/123456789/stock_alert_bot.png"
                }
                
                # Use rate limiter if enabled
                if self.rate_limiter:
                    success_send, status = self.rate_limiter.send_with_rate_limit(webhook_url, payload)
                    if success_send:
                        logger.info(f"Alert sent successfully to economy channel: {channel_name}")
                        channels_sent.append(channel_name)
                        success = True
                    else:
                        logger.error(f"Failed to send alert to {channel_name} (rate limited or error)")
                else:
                    # Original sending logic without rate limiting
                    response = requests.post(webhook_url, json=payload, timeout=10)

                    if response.status_code == 204:
                        logger.info(f"Alert sent successfully to economy channel: {channel_name}")
                        channels_sent.append(channel_name)
                        success = True
                    else:
                        logger.error(f"Failed to send alert to {channel_name}: {response.status_code} - {response.text}")
                    
        except Exception as e:
            logger.error(f"Error sending economy alert: {e}")
        
        # Second, send to any matching custom channels
        logger.info(f"Checking for custom channels for alert: {alert.get('ticker', 'Unknown')}")
        try:
            custom_channels = self.get_custom_channels_for_alert(alert)
            logger.info(f"Found {len(custom_channels)} matching custom channels")
            
            for custom_name, custom_info in custom_channels:
                try:
                    webhook_url = custom_info.get('webhook_url', '')
                    
                    if not webhook_url:
                        logger.warning(f"No webhook URL for custom channel {custom_name}")
                        continue
                    
                    # Send to custom channel with optional spacing
                    spacing_config = self.config.get('message_spacing', {})
                    if spacing_config.get('enabled', False):
                        divider_char = spacing_config.get('divider_char', '━')
                        divider_length = spacing_config.get('divider_length', 30)
                        divider = divider_char * divider_length
                        formatted_message = f"{divider}\n**{custom_info.get('channel_name', custom_name)}**\n{message}\n{divider}"
                    else:
                        formatted_message = f"**{custom_info.get('channel_name', custom_name)}**\n{message}"
                    
                    payload = {
                        "content": formatted_message,
                        "username": "Stock Alert Bot - Custom",
                        "avatar_url": "https://cdn.discordapp.com/attachments/123456789/stock_alert_bot.png"
                    }
                    
                    # Use rate limiter if enabled
                    if self.rate_limiter:
                        success_send, status = self.rate_limiter.send_with_rate_limit(webhook_url, payload)
                        if success_send:
                            logger.info(f"Alert sent successfully to custom channel: {custom_name}")
                            channels_sent.append(custom_info.get('channel_name', custom_name))
                            success = True
                        else:
                            logger.error(f"Failed to send alert to custom channel {custom_name} (rate limited or error)")
                    else:
                        # Original sending logic without rate limiting
                        response = requests.post(webhook_url, json=payload, timeout=10)

                        if response.status_code == 204:
                            logger.info(f"Alert sent successfully to custom channel: {custom_name}")
                            channels_sent.append(custom_info.get('channel_name', custom_name))
                            success = True
                        else:
                            logger.error(f"Failed to send alert to custom channel {custom_name}: {response.status_code}")
                        
                except Exception as e:
                    logger.error(f"Error sending to custom channel {custom_name}: {e}")
                    
        except Exception as e:
            logger.error(f"Error processing custom channels: {e}")
        
        # Log summary of where the alert was sent
        if channels_sent:
            logger.info(f"Alert sent to {len(channels_sent)} channels: {', '.join(channels_sent)}")
        else:
            logger.warning(f"Alert was not sent to any channels")

        return success
    
    def get_available_channels(self, timeframe: Optional[str] = None) -> List[Dict]:
        """
        Get list of available Discord channels and their configurations
        Optionally filtered by timeframe (daily/hourly)
        
        Returns:
            List of channel configurations
        """
        channels = []
        mapping, _ = self._get_channel_mapping(timeframe)
        for channel_name, config in mapping.items():
            webhook = config.get('webhook_url', '')
            channels.append({
                'name': channel_name,
                'channel_name': config.get('channel_name', channel_name),
                'description': config.get('description', ''),
                'configured': bool(webhook) and not webhook.startswith('YOUR_')
            })
        return channels
    
    def update_channel_config(self, channel_name: str, webhook_url: str, timeframe: Optional[str] = None) -> bool:
        """
        Update a channel's webhook URL
        
        Args:
            channel_name: Name of the channel to update
            webhook_url: New webhook URL
            timeframe: Optional timeframe ('daily' or 'hourly')
            
        Returns:
            True if updated successfully, False otherwise
        """
        try:
            mapping, mapping_key = self._get_channel_mapping(timeframe)

            if channel_name not in mapping:
                logger.error(f"Channel {channel_name} not found in configuration (timeframe={timeframe})")
                return False

            mapping[channel_name]['webhook_url'] = webhook_url

            # Persist configuration
            self.config[mapping_key][channel_name]['webhook_url'] = webhook_url
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Updated webhook URL for {channel_name} (timeframe={self._normalize_timeframe(timeframe)})")
            return True
        except Exception as e:
            logger.error(f"Error updating channel config: {e}")
            return False

# Global router instance
discord_router = DiscordEconomyRouter()

def get_rate_limit_status() -> Dict:
    """
    Get the current rate limiting status

    Returns:
        Dictionary with rate limit information
    """
    if discord_router.rate_limiter:
        return discord_router.rate_limiter.get_queue_status()
    return {"rate_limiting": "disabled"}

def send_economy_discord_alert(alert: Dict, message: str) -> bool:
    """
    Convenience function to send economy-based Discord alert
    
    Args:
        alert: Alert dictionary
        message: Alert message
        
    Returns:
        True if sent successfully, False otherwise
    """
    return discord_router.send_economy_alert(alert, message)

def get_stock_economy_classification(ticker: str) -> Optional[str]:
    """
    Convenience function to get stock economy classification
    
    Args:
        ticker: Stock ticker
        
    Returns:
        Economy classification or None
    """
    return discord_router.get_stock_economy(ticker)
