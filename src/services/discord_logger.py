"""
Asynchronous Discord Logger

Provides non-blocking Discord webhook logging using a background thread and queue.
This prevents Discord API calls from blocking the main application thread.

Functions:
- log_to_discord_async(message): Queue a message for async Discord logging
- flush_discord_logs(): Flush all queued messages to Discord
- shutdown_discord_logger(): Gracefully shutdown the background thread
"""

import os
import queue
import threading
import time
import requests
from typing import List, Optional
from dotenv import load_dotenv

from src.utils.discord_env import get_discord_environment_tag, is_discord_send_enabled

# Load environment variables
load_dotenv()


def _is_discord_logging_enabled() -> bool:
    """Return True only if DISCORD_LOGGING_ENABLED is explicitly true/1/yes (case-insensitive)."""
    val = (os.getenv("DISCORD_LOGGING_ENABLED") or "").strip().lower()
    return val in ("true", "1", "yes")


# Discord Configuration - must be set in environment
DISCORD_LOGGING_ENABLED = _is_discord_logging_enabled()
WEBHOOK_URL_LOGGING = os.getenv("WEBHOOK_URL_LOGGING")
WEBHOOK_URL_LOGGING_2 = os.getenv("WEBHOOK_URL_LOGGING_2")
MAX_DISCORD_MESSAGE_LENGTH = 2000
BATCH_DELAY = 0.5  # Delay in seconds before sending batched messages
MAX_BATCH_SIZE = 10  # Maximum number of messages to batch together

# Message Queue and Threading
_message_queue = queue.Queue()
_worker_thread: Optional[threading.Thread] = None
_shutdown_event = threading.Event()
_thread_lock = threading.Lock()


def split_message(message: str, max_length: int = MAX_DISCORD_MESSAGE_LENGTH) -> List[str]:
    """
    Split a long message into multiple chunks that fit Discord's message length limit.
    
    Args:
        message: The message to split
        max_length: Maximum length per chunk (default: 2000)
        
    Returns:
        List of message chunks
    """
    lines = message.split("\n")
    chunks = []
    current_chunk = ""
    
    for line in lines:
        # Reserve 6 characters for code block fences
        if len(current_chunk) + len(line) + 1 < max_length - 6:
            current_chunk += line + "\n"
        else:
            if current_chunk:
                chunks.append(f"```{current_chunk.strip()}```")
            current_chunk = line + "\n"
    
    if current_chunk:
        chunks.append(f"```{current_chunk.strip()}```")
    
    return chunks if chunks else [""]


def _send_to_discord(message: str) -> bool:
    """
    Send a message to Discord webhook(s).

    Args:
        message: Message to send

    Returns:
        True if sent successfully, False otherwise
    """
    if not is_discord_send_enabled() or not DISCORD_LOGGING_ENABLED:
        return False
    if not WEBHOOK_URL_LOGGING:
        return False
    
    # Check if webhook is configured
    if WEBHOOK_URL_LOGGING == "YOUR_WEBHOOK_URL_HERE" or "YOUR_" in WEBHOOK_URL_LOGGING:
        return False
    
    success = False
    
    # Split message if it's too long
    messages = split_message(message, MAX_DISCORD_MESSAGE_LENGTH)
    
    tag = get_discord_environment_tag()
    for msg in messages:
        payload = {"content": tag + msg}
        
        # Try primary webhook
        try:
            response = requests.post(WEBHOOK_URL_LOGGING, json=payload, timeout=10)
            if response.status_code == 204:
                success = True
            else:
                print(f"Discord webhook error: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Error sending to primary Discord webhook: {e}")
            
            # Try secondary webhook if primary fails
            if WEBHOOK_URL_LOGGING_2 and WEBHOOK_URL_LOGGING_2 != WEBHOOK_URL_LOGGING:
                try:
                    response_2 = requests.post(WEBHOOK_URL_LOGGING_2, json=payload, timeout=10)
                    if response_2.status_code == 204:
                        success = True
                except Exception as e2:
                    print(f"Error sending to secondary Discord webhook: {e2}")
        
        # Small delay between messages to avoid rate limiting
        if len(messages) > 1:
            time.sleep(0.5)
    
    return success


def _worker_loop():
    """
    Background worker thread that processes queued Discord messages.
    Batches multiple messages together to reduce API calls.
    """
    buffer = []
    last_send_time = time.time()
    
    while not _shutdown_event.is_set():
        try:
            # Try to get a message with a short timeout
            try:
                message = _message_queue.get(timeout=0.1)
                buffer.append(message)
                _message_queue.task_done()
            except queue.Empty:
                pass
            
            # Send if buffer is full or enough time has passed
            current_time = time.time()
            should_send = (
                len(buffer) >= MAX_BATCH_SIZE or
                (buffer and (current_time - last_send_time) >= BATCH_DELAY)
            )
            
            if should_send and buffer:
                # Combine messages with newlines
                combined_message = "\n".join(buffer)
                _send_to_discord(combined_message)
                buffer.clear()
                last_send_time = current_time
                
        except Exception as e:
            print(f"Error in Discord logger worker thread: {e}")
            buffer.clear()  # Clear buffer on error to prevent infinite loop
    
    # Send any remaining messages before shutdown
    if buffer:
        combined_message = "\n".join(buffer)
        _send_to_discord(combined_message)


def _ensure_worker_thread():
    """
    Ensure the background worker thread is running.
    Thread-safe initialization.
    """
    global _worker_thread
    
    with _thread_lock:
        if _worker_thread is None or not _worker_thread.is_alive():
            _shutdown_event.clear()
            _worker_thread = threading.Thread(target=_worker_loop, daemon=True, name="DiscordLogger")
            _worker_thread.start()


def log_to_discord_async(message: str):
    """
    Queue a message for asynchronous Discord logging.
    This function returns immediately without blocking.

    Args:
        message: Message to log to Discord
    """
    if not message or not is_discord_send_enabled() or not DISCORD_LOGGING_ENABLED:
        return

    # Ensure worker thread is running
    _ensure_worker_thread()

    # Queue the message
    _message_queue.put(str(message))


def flush_discord_logs():
    """
    Flush all queued Discord log messages.
    Blocks until all messages in the queue have been sent.
    """
    # Ensure worker thread is running
    _ensure_worker_thread()
    
    # Wait for queue to be empty
    _message_queue.join()


def shutdown_discord_logger():
    """
    Gracefully shutdown the Discord logger background thread.
    Flushes all pending messages before shutting down.
    """
    global _worker_thread
    
    # Signal shutdown
    _shutdown_event.set()
    
    # Wait for thread to finish
    if _worker_thread and _worker_thread.is_alive():
        _worker_thread.join(timeout=5.0)
    
    _worker_thread = None


# Ensure clean shutdown on exit
import atexit
atexit.register(shutdown_discord_logger)
