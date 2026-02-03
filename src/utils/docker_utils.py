"""
Docker container management utilities for scheduler status pages.

Provides functions to check container status and start/stop containers
when running in a Docker environment.
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Container name mappings
CONTAINER_NAMES = {
    "hourly": "stockalert-hourly-scheduler",
    "daily": "stockalert-daily-scheduler",
    "futures": "stockalert-futures-scheduler",
}

# Service name mappings (for docker-compose commands)
SERVICE_NAMES = {
    "hourly": "hourly-scheduler",
    "daily": "daily-scheduler",
    "futures": "futures-scheduler",
}

# Docker compose project name (derived from directory name)
COMPOSE_PROJECT_NAME = os.getenv("COMPOSE_PROJECT_NAME", "stock-market-alert-app")


def is_running_in_docker() -> bool:
    """Check if the current process is running inside a Docker container."""
    # Check for .dockerenv file
    if os.path.exists("/.dockerenv"):
        return True
    # Check cgroup for docker
    try:
        with open("/proc/1/cgroup", "r") as f:
            return "docker" in f.read()
    except (FileNotFoundError, PermissionError):
        pass
    # Check for DOCKER_CONTAINER env var (can be set in docker-compose)
    if os.getenv("DOCKER_CONTAINER"):
        return True
    return False


def _run_docker_command(args: list[str], timeout: int = 30) -> Tuple[bool, str]:
    """
    Run a docker command and return (success, output).

    Args:
        args: Command arguments (e.g., ["ps", "-q", "-f", "name=container"])
        timeout: Command timeout in seconds

    Returns:
        Tuple of (success: bool, output: str)
    """
    try:
        result = subprocess.run(
            ["docker"] + args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout.strip()
        if result.returncode == 0:
            return True, output
        return False, result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except FileNotFoundError:
        return False, "Docker command not found"
    except Exception as exc:
        return False, str(exc)


def _run_compose_command(args: list[str], timeout: int = 60) -> Tuple[bool, str]:
    """
    Run a docker-compose command and return (success, output).

    Tries 'docker compose' first (Docker Compose V2), then 'docker-compose' (V1).

    Args:
        args: Command arguments (e.g., ["up", "-d", "service-name"])
        timeout: Command timeout in seconds

    Returns:
        Tuple of (success: bool, output: str)
    """
    # Try Docker Compose V2 first
    try:
        result = subprocess.run(
            ["docker", "compose"] + args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
        # If V2 failed, try V1
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    except Exception:
        pass

    # Try Docker Compose V1
    try:
        result = subprocess.run(
            ["docker-compose"] + args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout.strip()
        if result.returncode == 0:
            return True, output
        return False, result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except FileNotFoundError:
        return False, "docker-compose command not found"
    except Exception as exc:
        return False, str(exc)


def is_container_running(scheduler_type: str) -> bool:
    """
    Check if a scheduler container is running.

    Args:
        scheduler_type: One of "hourly", "daily", or "futures"

    Returns:
        True if the container is running, False otherwise
    """
    container_name = CONTAINER_NAMES.get(scheduler_type)
    if not container_name:
        logger.warning(f"Unknown scheduler type: {scheduler_type}")
        return False

    success, output = _run_docker_command(
        ["ps", "-q", "-f", f"name={container_name}", "-f", "status=running"]
    )
    return success and bool(output)


def get_container_status(scheduler_type: str) -> Optional[str]:
    """
    Get the status of a scheduler container.

    Args:
        scheduler_type: One of "hourly", "daily", or "futures"

    Returns:
        Container status string (e.g., "running", "exited", "created") or None
    """
    container_name = CONTAINER_NAMES.get(scheduler_type)
    if not container_name:
        return None

    success, output = _run_docker_command(
        ["ps", "-a", "--format", "{{.Status}}", "-f", f"name={container_name}"]
    )
    if success and output:
        # Parse status like "Up 2 hours" or "Exited (0) 5 minutes ago"
        status_lower = output.lower()
        if status_lower.startswith("up"):
            return "running"
        elif "exited" in status_lower:
            return "exited"
        elif "created" in status_lower:
            return "created"
        elif "restarting" in status_lower:
            return "restarting"
        return output
    return None


def get_container_health(scheduler_type: str) -> Optional[str]:
    """
    Get the health status of a scheduler container.

    Args:
        scheduler_type: One of "hourly", "daily", or "futures"

    Returns:
        Health status string or None if not available
    """
    container_name = CONTAINER_NAMES.get(scheduler_type)
    if not container_name:
        return None

    success, output = _run_docker_command(
        ["inspect", "--format", "{{.State.Health.Status}}", container_name]
    )
    if success and output and output != "<no value>":
        return output
    return None


def start_container(scheduler_type: str) -> Tuple[bool, str]:
    """
    Start a scheduler container.

    Uses direct docker commands (docker start) which work from inside containers
    without needing access to docker-compose.yml.

    Args:
        scheduler_type: One of "hourly", "daily", or "futures"

    Returns:
        Tuple of (success: bool, message: str)
    """
    container_name = CONTAINER_NAMES.get(scheduler_type)
    if not container_name:
        return False, f"Unknown scheduler type: {scheduler_type}"

    # Check if already running
    if is_container_running(scheduler_type):
        return True, "Container is already running"

    # Try to start existing container
    success, output = _run_docker_command(["start", container_name], timeout=60)
    if success:
        return True, f"Started {container_name}"

    # If container doesn't exist, we can't start it from here
    # User needs to run docker-compose up from host
    return False, f"Container {container_name} not found. Run 'docker compose up -d {SERVICE_NAMES.get(scheduler_type)}' from host."


def stop_container(scheduler_type: str) -> Tuple[bool, str]:
    """
    Stop a scheduler container.

    Uses direct docker commands (docker stop) which work from inside containers.

    Args:
        scheduler_type: One of "hourly", "daily", or "futures"

    Returns:
        Tuple of (success: bool, message: str)
    """
    container_name = CONTAINER_NAMES.get(scheduler_type)
    if not container_name:
        return False, f"Unknown scheduler type: {scheduler_type}"

    # Check if already stopped
    if not is_container_running(scheduler_type):
        return True, "Container is already stopped"

    success, output = _run_docker_command(["stop", container_name], timeout=60)
    if success:
        return True, f"Stopped {container_name}"
    return False, f"Failed to stop {container_name}: {output}"


def restart_container(scheduler_type: str) -> Tuple[bool, str]:
    """
    Restart a scheduler container.

    Uses direct docker commands (docker restart) which work from inside containers.

    Args:
        scheduler_type: One of "hourly", "daily", or "futures"

    Returns:
        Tuple of (success: bool, message: str)
    """
    container_name = CONTAINER_NAMES.get(scheduler_type)
    if not container_name:
        return False, f"Unknown scheduler type: {scheduler_type}"

    success, output = _run_docker_command(["restart", container_name], timeout=120)
    if success:
        return True, f"Restarted {container_name}"
    return False, f"Failed to restart {container_name}: {output}"


def get_container_logs(scheduler_type: str, lines: int = 50) -> str:
    """
    Get recent logs from a scheduler container.

    Args:
        scheduler_type: One of "hourly", "daily", or "futures"
        lines: Number of log lines to retrieve

    Returns:
        Log output string
    """
    container_name = CONTAINER_NAMES.get(scheduler_type)
    if not container_name:
        return f"Unknown scheduler type: {scheduler_type}"

    success, output = _run_docker_command(
        ["logs", "--tail", str(lines), container_name],
        timeout=30,
    )
    if success:
        return output
    return f"Failed to get logs: {output}"
