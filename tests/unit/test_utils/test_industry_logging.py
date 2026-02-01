"""
Unit tests for the industry_logging module.

Tests the industry-specific logging functionality including:
- Alert check logging with different levels
- Data quality issue logging
- Industry validation logging
- Discord integration (with mocking)
- Error handling
"""

import logging
import logging.handlers
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock, call
import pytest

from src.utils.industry_logging import (
    log_alert_check,
    log_data_quality_issue,
    log_industry_validation,
)


class TestLogAlertCheck:
    """Tests for the log_alert_check function."""

    def setup_method(self):
        """Setup before each test."""
        # Create a logger for testing
        self.logger = logging.getLogger("src.utils.industry_logging")
        self.original_level = self.logger.level
        self.logger.setLevel(logging.DEBUG)
        
        # Add handler to capture logs
        self.log_handler = logging.handlers.MemoryHandler(capacity=1000)
        self.logger.addHandler(self.log_handler)

    def teardown_method(self):
        """Cleanup after each test."""
        self.logger.removeHandler(self.log_handler)
        self.logger.setLevel(self.original_level)

    @patch("src.utils.industry_logging.logger")
    def test_info_level_logging(self, mock_logger):
        """Test logging at info level."""
        log_alert_check("AAPL", "Data Check", "Test message", level="info")
        
        # Should call logger.info
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        
        # Verify message format
        assert "[AAPL]" in call_args
        assert "[Data Check]" in call_args
        assert "Test message" in call_args

    @patch("src.utils.industry_logging.logger")
    def test_warning_level_logging(self, mock_logger):
        """Test logging at warning level."""
        log_alert_check("TSLA", "Validation", "Warning message", level="warning")
        
        # Should call logger.warning
        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args[0][0]
        
        assert "[TSLA]" in call_args
        assert "[Validation]" in call_args
        assert "Warning message" in call_args

    @patch("src.utils.industry_logging.logger")
    def test_error_level_logging(self, mock_logger):
        """Test logging at error level."""
        log_alert_check("GOOGL", "API Error", "Error message", level="error")
        
        # Should call logger.error
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args[0][0]
        
        assert "[GOOGL]" in call_args
        assert "[API Error]" in call_args
        assert "Error message" in call_args

    @patch("src.utils.industry_logging.logger")
    def test_debug_level_logging(self, mock_logger):
        """Test logging at debug level."""
        log_alert_check("MSFT", "Debug Check", "Debug message", level="debug")
        
        # Should call logger.debug
        mock_logger.debug.assert_called_once()
        call_args = mock_logger.debug.call_args[0][0]
        
        assert "[MSFT]" in call_args
        assert "[Debug Check]" in call_args
        assert "Debug message" in call_args

    @patch("src.utils.industry_logging.logger")
    def test_default_level_is_info(self, mock_logger):
        """Test that default log level is info."""
        log_alert_check("AMZN", "Check", "Message")
        
        # Should default to info level
        mock_logger.info.assert_called_once()
        mock_logger.warning.assert_not_called()
        mock_logger.error.assert_not_called()

    @patch("src.utils.industry_logging.logger")
    def test_timestamp_in_message(self, mock_logger):
        """Test that timestamp is included in formatted message."""
        log_alert_check("AAPL", "Check", "Message")
        
        call_args = mock_logger.info.call_args[0][0]
        
        # Should contain a timestamp in format YYYY-MM-DD HH:MM:SS
        # Check for date pattern
        assert "]" in call_args  # Timestamp should be in brackets
        # Extract the first part (timestamp)
        timestamp_part = call_args.split("]")[0].strip("[")
        # Should have format like "2024-01-31 12:34:56"
        assert len(timestamp_part) == 19  # YYYY-MM-DD HH:MM:SS

    @patch("src.utils.industry_logging.logger")
    def test_message_format_structure(self, mock_logger):
        """Test the complete message format structure."""
        log_alert_check("TEST", "Type", "Content")
        
        call_args = mock_logger.info.call_args[0][0]
        
        # Format should be: [timestamp] [ticker] [check_type] message
        parts = call_args.split("]")
        assert len(parts) >= 4  # timestamp, ticker, check_type, and message
        
        # Check ticker is second bracket
        assert "[TEST]" in call_args
        assert "[Type]" in call_args
        assert "Content" in call_args

    @patch("src.utils.industry_logging.logger")
    @patch("src.services.discord_logger.log_to_discord_async")
    def test_discord_logging_for_warning(self, mock_discord, mock_logger):
        """Test that warnings are sent to Discord."""
        log_alert_check("AAPL", "Alert", "Warning message", level="warning")
        
        # Should call Discord logger
        mock_discord.assert_called_once()
        discord_call_args = mock_discord.call_args[0][0]
        
        # Verify Discord message format
        assert "Alert" in discord_call_args
        assert "AAPL" in discord_call_args
        assert "Warning message" in discord_call_args

    @patch("src.utils.industry_logging.logger")
    @patch("src.services.discord_logger.log_to_discord_async")
    def test_discord_logging_for_error(self, mock_discord, mock_logger):
        """Test that errors are sent to Discord."""
        log_alert_check("TSLA", "Critical", "Error message", level="error")
        
        # Should call Discord logger
        mock_discord.assert_called_once()
        discord_call_args = mock_discord.call_args[0][0]
        
        assert "Critical" in discord_call_args
        assert "TSLA" in discord_call_args
        assert "Error message" in discord_call_args

    @patch("src.utils.industry_logging.logger")
    @patch("src.services.discord_logger.log_to_discord_async")
    def test_no_discord_logging_for_info(self, mock_discord, mock_logger):
        """Test that info level messages are not sent to Discord."""
        log_alert_check("MSFT", "Info", "Info message", level="info")
        
        # Should not call Discord logger
        mock_discord.assert_not_called()

    @patch("src.utils.industry_logging.logger")
    @patch("src.services.discord_logger.log_to_discord_async")
    def test_no_discord_logging_for_debug(self, mock_discord, mock_logger):
        """Test that debug level messages are not sent to Discord."""
        log_alert_check("GOOGL", "Debug", "Debug message", level="debug")
        
        # Should not call Discord logger
        mock_discord.assert_not_called()

    @patch("src.utils.industry_logging.logger")
    @patch("src.services.discord_logger.log_to_discord_async")
    def test_discord_import_error_handled(self, mock_discord, mock_logger):
        """Test that Discord import errors are handled gracefully."""
        # Make Discord import fail
        mock_discord.side_effect = ImportError("Discord module not available")
        
        # Should not raise exception
        log_alert_check("AAPL", "Check", "Message", level="warning")
        
        # Standard logger should still be called
        mock_logger.warning.assert_called_once()

    @patch("src.utils.industry_logging.logger")
    @patch("src.services.discord_logger.log_to_discord_async")
    def test_discord_exception_handled(self, mock_discord, mock_logger):
        """Test that Discord exceptions are handled gracefully."""
        # Make Discord raise exception
        mock_discord.side_effect = Exception("Discord connection failed")
        
        # Should not raise exception
        log_alert_check("TSLA", "Alert", "Message", level="error")
        
        # Standard logger should still be called
        mock_logger.error.assert_called_once()

    @patch("src.utils.industry_logging.logger")
    def test_case_insensitive_log_level(self, mock_logger):
        """Test that log level is case insensitive."""
        log_alert_check("AAPL", "Check", "Message", level="INFO")
        mock_logger.info.assert_called_once()
        
        mock_logger.reset_mock()
        
        log_alert_check("AAPL", "Check", "Message", level="WARNING")
        mock_logger.warning.assert_called_once()
        
        mock_logger.reset_mock()
        
        log_alert_check("AAPL", "Check", "Message", level="ERROR")
        mock_logger.error.assert_called_once()

    @patch("src.utils.industry_logging.logger")
    def test_unknown_log_level_defaults_to_info(self, mock_logger):
        """Test that unknown log levels default to info."""
        log_alert_check("AAPL", "Check", "Message", level="unknown")
        
        # Should default to info
        mock_logger.info.assert_called_once()
        mock_logger.warning.assert_not_called()
        mock_logger.error.assert_not_called()

    @patch("src.utils.industry_logging.logger")
    def test_empty_ticker(self, mock_logger):
        """Test handling of empty ticker."""
        log_alert_check("", "Check", "Message")
        
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        assert "[]" in call_args  # Empty ticker brackets

    @patch("src.utils.industry_logging.logger")
    def test_empty_check_type(self, mock_logger):
        """Test handling of empty check type."""
        log_alert_check("AAPL", "", "Message")
        
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        assert "[]" in call_args  # Empty check type brackets

    @patch("src.utils.industry_logging.logger")
    def test_empty_message(self, mock_logger):
        """Test handling of empty message."""
        log_alert_check("AAPL", "Check", "")
        
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        assert "[AAPL]" in call_args
        assert "[Check]" in call_args

    @patch("src.utils.industry_logging.logger")
    def test_special_characters_in_message(self, mock_logger):
        """Test handling of special characters in message."""
        special_message = "Price: $123.45, Change: +5.2%, Volume: 1,234,567"
        log_alert_check("AAPL", "Check", special_message)
        
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        assert special_message in call_args

    @patch("src.utils.industry_logging.logger")
    def test_multiline_message(self, mock_logger):
        """Test handling of multiline messages."""
        multiline_message = "Line 1\nLine 2\nLine 3"
        log_alert_check("AAPL", "Check", multiline_message)
        
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        assert "Line 1" in call_args
        assert "Line 2" in call_args
        assert "Line 3" in call_args


class TestLogDataQualityIssue:
    """Tests for the log_data_quality_issue function."""

    @patch("src.utils.industry_logging.log_alert_check")
    def test_calls_log_alert_check_with_correct_params(self, mock_log_alert):
        """Test that log_data_quality_issue calls log_alert_check correctly."""
        log_data_quality_issue("AAPL", "Stale Data", "Data is 3 days old")
        
        mock_log_alert.assert_called_once_with(
            "AAPL",
            "Data Quality - Stale Data",
            "Data is 3 days old",
            level="warning"
        )

    @patch("src.utils.industry_logging.log_alert_check")
    def test_default_severity_is_warning(self, mock_log_alert):
        """Test that default severity level is warning."""
        log_data_quality_issue("TSLA", "Missing Data", "Volume missing")
        
        call_args = mock_log_alert.call_args
        assert call_args[1]["level"] == "warning"

    @patch("src.utils.industry_logging.log_alert_check")
    def test_custom_severity_info(self, mock_log_alert):
        """Test data quality issue with info severity."""
        log_data_quality_issue("GOOGL", "Minor Issue", "Details", severity="info")
        
        mock_log_alert.assert_called_once()
        call_args = mock_log_alert.call_args
        assert call_args[1]["level"] == "info"

    @patch("src.utils.industry_logging.log_alert_check")
    def test_custom_severity_error(self, mock_log_alert):
        """Test data quality issue with error severity."""
        log_data_quality_issue("MSFT", "Critical Issue", "Details", severity="error")
        
        mock_log_alert.assert_called_once()
        call_args = mock_log_alert.call_args
        assert call_args[1]["level"] == "error"

    @patch("src.utils.industry_logging.log_alert_check")
    def test_check_type_format(self, mock_log_alert):
        """Test that check_type is formatted correctly."""
        log_data_quality_issue("AMZN", "Stale Data", "Details")
        
        call_args = mock_log_alert.call_args[0]
        check_type = call_args[1]
        
        # Should be prefixed with "Data Quality - "
        assert check_type == "Data Quality - Stale Data"

    @patch("src.utils.industry_logging.log_alert_check")
    def test_various_issue_types(self, mock_log_alert):
        """Test different types of data quality issues."""
        issue_types = [
            "Stale Data",
            "Missing Data",
            "Invalid Data",
            "Corrupted Data",
            "Out of Range"
        ]
        
        for issue_type in issue_types:
            mock_log_alert.reset_mock()
            log_data_quality_issue("TEST", issue_type, "Details")
            
            call_args = mock_log_alert.call_args[0]
            assert f"Data Quality - {issue_type}" in call_args


class TestLogIndustryValidation:
    """Tests for the log_industry_validation function."""

    @patch("src.utils.industry_logging.log_alert_check")
    def test_passed_validation_info_level(self, mock_log_alert):
        """Test that passed validation uses info level."""
        log_industry_validation("AAPL", "Technology", "Validation passed", passed=True)
        
        mock_log_alert.assert_called_once()
        call_args = mock_log_alert.call_args
        assert call_args[1]["level"] == "info"

    @patch("src.utils.industry_logging.log_alert_check")
    def test_failed_validation_warning_level(self, mock_log_alert):
        """Test that failed validation uses warning level."""
        log_industry_validation("XYZ", "Unknown", "Industry not found", passed=False)
        
        mock_log_alert.assert_called_once()
        call_args = mock_log_alert.call_args
        assert call_args[1]["level"] == "warning"

    @patch("src.utils.industry_logging.log_alert_check")
    def test_default_passed_is_true(self, mock_log_alert):
        """Test that default passed value is True."""
        log_industry_validation("TSLA", "Automotive", "Result")
        
        call_args = mock_log_alert.call_args
        assert call_args[1]["level"] == "info"

    @patch("src.utils.industry_logging.log_alert_check")
    def test_passed_validation_check_mark(self, mock_log_alert):
        """Test that passed validation includes check mark."""
        log_industry_validation("GOOGL", "Technology", "Valid", passed=True)
        
        call_args = mock_log_alert.call_args[0]
        message = call_args[2]
        
        assert "✓" in message

    @patch("src.utils.industry_logging.log_alert_check")
    def test_failed_validation_x_mark(self, mock_log_alert):
        """Test that failed validation includes X mark."""
        log_industry_validation("ABC", "Unknown", "Invalid", passed=False)
        
        call_args = mock_log_alert.call_args[0]
        message = call_args[2]
        
        assert "✗" in message

    @patch("src.utils.industry_logging.log_alert_check")
    def test_industry_in_message(self, mock_log_alert):
        """Test that industry is included in message."""
        log_industry_validation("MSFT", "Technology", "Result")
        
        call_args = mock_log_alert.call_args[0]
        message = call_args[2]
        
        assert "[Technology]" in message

    @patch("src.utils.industry_logging.log_alert_check")
    def test_validation_result_in_message(self, mock_log_alert):
        """Test that validation result is included in message."""
        result_text = "Sector metadata validated successfully"
        log_industry_validation("AMZN", "E-commerce", result_text)
        
        call_args = mock_log_alert.call_args[0]
        message = call_args[2]
        
        assert result_text in message

    @patch("src.utils.industry_logging.log_alert_check")
    def test_check_type_is_industry_validation(self, mock_log_alert):
        """Test that check_type is 'Industry Validation'."""
        log_industry_validation("AAPL", "Technology", "Result")
        
        call_args = mock_log_alert.call_args[0]
        check_type = call_args[1]
        
        assert check_type == "Industry Validation"

    @patch("src.utils.industry_logging.log_alert_check")
    def test_message_format_passed(self, mock_log_alert):
        """Test complete message format for passed validation."""
        log_industry_validation("TSLA", "Automotive", "All checks passed", passed=True)
        
        call_args = mock_log_alert.call_args[0]
        message = call_args[2]
        
        # Should be: ✓ [Industry] validation_result
        assert message == "✓ [Automotive] All checks passed"

    @patch("src.utils.industry_logging.log_alert_check")
    def test_message_format_failed(self, mock_log_alert):
        """Test complete message format for failed validation."""
        log_industry_validation("XYZ", "Unknown", "No data found", passed=False)
        
        call_args = mock_log_alert.call_args[0]
        message = call_args[2]
        
        # Should be: ✗ [Industry] validation_result
        assert message == "✗ [Unknown] No data found"

    @patch("src.utils.industry_logging.log_alert_check")
    def test_various_industries(self, mock_log_alert):
        """Test validation with various industry types."""
        industries = [
            "Technology",
            "Finance",
            "Healthcare",
            "Energy",
            "Consumer Goods",
            "Unknown"
        ]
        
        for industry in industries:
            mock_log_alert.reset_mock()
            log_industry_validation("TEST", industry, "Result")
            
            call_args = mock_log_alert.call_args[0]
            message = call_args[2]
            assert f"[{industry}]" in message


class TestIntegration:
    """Integration tests for industry_logging module."""

    @patch("src.utils.industry_logging.logger")
    @patch("src.services.discord_logger.log_to_discord_async")
    def test_data_quality_issue_with_warning_triggers_discord(
        self, mock_discord, mock_logger
    ):
        """Test that data quality warnings trigger Discord logging."""
        log_data_quality_issue("AAPL", "Stale Data", "Data is old", severity="warning")
        
        # Should log to both standard logger and Discord
        mock_logger.warning.assert_called_once()
        mock_discord.assert_called_once()

    @patch("src.utils.industry_logging.logger")
    @patch("src.services.discord_logger.log_to_discord_async")
    def test_data_quality_issue_with_error_triggers_discord(
        self, mock_discord, mock_logger
    ):
        """Test that data quality errors trigger Discord logging."""
        log_data_quality_issue("TSLA", "Missing Data", "Critical", severity="error")
        
        # Should log to both standard logger and Discord
        mock_logger.error.assert_called_once()
        mock_discord.assert_called_once()

    @patch("src.utils.industry_logging.logger")
    @patch("src.services.discord_logger.log_to_discord_async")
    def test_failed_industry_validation_triggers_discord(
        self, mock_discord, mock_logger
    ):
        """Test that failed industry validation triggers Discord logging."""
        log_industry_validation("XYZ", "Unknown", "Not found", passed=False)
        
        # Should log to both standard logger and Discord (warning level)
        mock_logger.warning.assert_called_once()
        mock_discord.assert_called_once()

    @patch("src.utils.industry_logging.logger")
    @patch("src.services.discord_logger.log_to_discord_async")
    def test_passed_industry_validation_no_discord(
        self, mock_discord, mock_logger
    ):
        """Test that passed industry validation does not trigger Discord."""
        log_industry_validation("AAPL", "Technology", "Valid", passed=True)
        
        # Should only log to standard logger (info level)
        mock_logger.info.assert_called_once()
        mock_discord.assert_not_called()

    @patch("src.utils.industry_logging.logger")
    def test_multiple_sequential_calls(self, mock_logger):
        """Test multiple sequential logging calls."""
        log_alert_check("AAPL", "Check1", "Message1")
        log_alert_check("TSLA", "Check2", "Message2")
        log_data_quality_issue("GOOGL", "Issue", "Details")
        log_industry_validation("MSFT", "Tech", "Valid")
        
        # Should have 4 total log calls (all at info/warning level)
        assert mock_logger.info.call_count == 3  # 2 alert checks + 1 validation
        assert mock_logger.warning.call_count == 1  # 1 data quality issue
