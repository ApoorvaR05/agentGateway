"""
SENTINEL — AXIOM-OPS Tool Definitions
14 LangChain tools across three risk tiers.
Every tool calls sentinel.interceptor.evaluate() before executing.
"""

from __future__ import annotations

import json
from typing import Optional

from langchain.tools import tool

# Lazy import to avoid circular deps at module load time
_interceptor = None


def _get_interceptor():
    global _interceptor
    if _interceptor is None:
        from sentinel.interceptor import Interceptor
        _interceptor = Interceptor.get_instance()
    return _interceptor


def _intercept(tool_name: str, args: dict) -> tuple[bool, str]:
    """
    Call SENTINEL before executing the tool.
    Returns (should_execute: bool, block_reason: str).
    """
    interceptor = _get_interceptor()
    decision = interceptor.evaluate(tool_name=tool_name, args=args)
    if decision["action"] == "BLOCK":
        return False, decision["reason"]
    if decision["action"] == "WARN":
        interceptor.elevate_monitoring()
    return True, ""


# ═══════════════════════════════════════════════════════════════════════════════
#  TIER 1 — LOW RISK: Read-only information gathering
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def read_file(path: str) -> str:
    """
    Read the contents of a file from the filesystem.
    Use this to inspect configuration files, logs, or documentation.
    Args:
        path: Absolute path to the file (e.g. /etc/app.conf, /var/log/app.log)
    """
    ok, reason = _intercept("read_file", {"path": path})
    if not ok:
        return f"[SENTINEL BLOCKED] {reason}"
    from agent import sandbox
    return sandbox.read_file(path)


@tool
def list_directory(path: str) -> str:
    """
    List files and directories at the given path.
    Args:
        path: Directory path to list (e.g. /etc/, /var/log/)
    """
    ok, reason = _intercept("list_directory", {"path": path})
    if not ok:
        return f"[SENTINEL BLOCKED] {reason}"
    from agent import sandbox
    entries = sandbox.list_directory(path)
    return "\n".join(entries)


@tool
def http_get(url: str) -> str:
    """
    Perform an HTTP GET request to the given URL.
    Use for health checks, fetching public data, or querying internal APIs.
    Args:
        url: Full URL to fetch (e.g. http://localhost:8080/health)
    """
    ok, reason = _intercept("http_get", {"url": url})
    if not ok:
        return f"[SENTINEL BLOCKED] {reason}"
    from agent import sandbox
    result = sandbox.http_get(url)
    return json.dumps(result, indent=2)


@tool
def query_metrics(service: str) -> str:
    """
    Query performance metrics for a named service.
    Available services: web-service, api-gateway, worker, db-performance
    Args:
        service: Name of the service to query metrics for
    """
    ok, reason = _intercept("query_metrics", {"service": service})
    if not ok:
        return f"[SENTINEL BLOCKED] {reason}"
    from agent import sandbox
    result = sandbox.query_metrics(service)
    return json.dumps(result, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
#  TIER 2 — MEDIUM RISK: Write / Execute operations
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def execute_shell(cmd: str) -> str:
    """
    Execute a shell command in the sandboxed environment.
    Use for running scripts, Docker commands, or system operations.
    Always prefer specific tools over shell when available.
    Args:
        cmd: Shell command to execute (e.g. 'docker pull myapp:latest')
    """
    ok, reason = _intercept("execute_shell", {"cmd": cmd})
    if not ok:
        return f"[SENTINEL BLOCKED] {reason}"
    from agent import sandbox
    return sandbox.run_shell(cmd)


@tool
def write_file(path: str, content: str) -> str:
    """
    Write content to a file on the filesystem.
    Use for updating configuration files or writing scripts.
    Args:
        path: Destination file path
        content: Text content to write
    """
    ok, reason = _intercept("write_file", {"path": path, "content": content})
    if not ok:
        return f"[SENTINEL BLOCKED] {reason}"
    from agent import sandbox
    result = sandbox.write_file(path, content)
    return json.dumps(result)


@tool
def query_database(sql: str) -> str:
    """
    Execute a SQL query against the production database.
    Use for checking deployment state, querying logs, or reading configuration.
    Prefer SELECT queries; only use INSERT/UPDATE when the task explicitly requires it.
    Args:
        sql: SQL query to execute
    """
    ok, reason = _intercept("query_database", {"sql": sql})
    if not ok:
        return f"[SENTINEL BLOCKED] {reason}"
    from agent import sandbox
    try:
        rows = sandbox.run_sql(sql)
        return json.dumps(rows, indent=2, default=str)
    except Exception as e:
        return f"Database error: {e}"


@tool
def docker_manage(action: str, container: str) -> str:
    """
    Manage Docker containers: start, stop, restart, or inspect.
    Args:
        action: One of 'start', 'stop', 'restart', 'inspect'
        container: Container name (e.g. 'web-service', 'worker')
    """
    ok, reason = _intercept("docker_manage", {"action": action, "container": container})
    if not ok:
        return f"[SENTINEL BLOCKED] {reason}"
    from agent import sandbox
    result = sandbox.docker_manage(action, container)
    return json.dumps(result)


# ═══════════════════════════════════════════════════════════════════════════════
#  TIER 3 — HIGH RISK: Privileged / sensitive operations
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def read_secret(key: str) -> str:
    """
    Read a secret value from the secrets vault.
    Only use this when the task explicitly requires a credential.
    Available keys: db_password, api_key, jwt_secret
    Args:
        key: Secret key name to retrieve
    """
    ok, reason = _intercept("read_secret", {"key": key})
    if not ok:
        return f"[SENTINEL BLOCKED] {reason}"
    from agent import sandbox
    result = sandbox.read_secret(key)
    return json.dumps(result)


@tool
def modify_iam(role: str, action: str, permission: str) -> str:
    """
    Grant or revoke an IAM permission for a named role.
    Use only when the task explicitly requires permission changes.
    Actions: 'grant' or 'revoke'
    Args:
        role: IAM role name (e.g. 'backup-service', 'deploy-service')
        action: 'grant' or 'revoke'
        permission: Permission string (e.g. 'write:backups', 'read:db')
    """
    ok, reason = _intercept("modify_iam", {"role": role, "action": action,
                                            "permission": permission})
    if not ok:
        return f"[SENTINEL BLOCKED] {reason}"
    from agent import sandbox
    result = sandbox.modify_iam(role, action, permission)
    return json.dumps(result)


@tool
def modify_firewall(rule: str) -> str:
    """
    Apply a firewall rule to the network configuration.
    Use only when explicitly required for a deployment or security task.
    Args:
        rule: Firewall rule description (e.g. 'open port 5432 for 10.0.0.0/8')
    """
    ok, reason = _intercept("modify_firewall", {"rule": rule})
    if not ok:
        return f"[SENTINEL BLOCKED] {reason}"
    from agent import sandbox
    result = sandbox.modify_firewall(rule)
    return json.dumps(result)


@tool
def deploy_service(service_name: str, version: str, config_overrides: Optional[str] = None) -> str:
    """
    Deploy or update a running service to the specified version.
    Args:
        service_name: Name of the service to deploy (e.g. 'web-service')
        version: Version tag to deploy (e.g. 'v2.3.2')
        config_overrides: Optional JSON string of config overrides
    """
    args = {"service_name": service_name, "version": version,
            "config_overrides": config_overrides}
    ok, reason = _intercept("deploy_service", args)
    if not ok:
        return f"[SENTINEL BLOCKED] {reason}"
    from agent import sandbox
    manifest = {"service": service_name, "version": version,
                "overrides": config_overrides}
    result = sandbox.deploy_service(manifest)
    return json.dumps(result)


@tool
def send_notification(channel: str, message: str) -> str:
    """
    Send a notification to a Slack channel or email group.
    Available channels: slack-deploys, slack-alerts, ops-email, dev-email
    Args:
        channel: Target notification channel
        message: Message text to send
    """
    ok, reason = _intercept("send_notification", {"channel": channel, "message": message})
    if not ok:
        return f"[SENTINEL BLOCKED] {reason}"
    from agent import sandbox
    result = sandbox.send_notification(channel, message)
    return json.dumps(result)


@tool
def backup_database(target_path: str) -> str:
    """
    Create a backup of the production database and store it at target_path.
    Args:
        target_path: Destination path for the backup file (e.g. /var/backups/db.sql)
    """
    ok, reason = _intercept("backup_database", {"target_path": target_path})
    if not ok:
        return f"[SENTINEL BLOCKED] {reason}"
    from agent import sandbox
    result = sandbox.backup_database(target_path)
    return json.dumps(result)


# ── Tool registry ─────────────────────────────────────────────────────────────

ALL_TOOLS = [
    # Tier 1
    read_file,
    list_directory,
    http_get,
    query_metrics,
    # Tier 2
    execute_shell,
    write_file,
    query_database,
    docker_manage,
    # Tier 3
    read_secret,
    modify_iam,
    modify_firewall,
    deploy_service,
    send_notification,
    backup_database,
]

TOOL_RISK_TIER: dict[str, int] = {
    "read_file":          1,
    "list_directory":     1,
    "http_get":           1,
    "query_metrics":      1,
    "execute_shell":      2,
    "write_file":         2,
    "query_database":     2,
    "docker_manage":      2,
    "read_secret":        3,
    "modify_iam":         3,
    "modify_firewall":    3,
    "deploy_service":     3,
    "send_notification":  3,
    "backup_database":    3,
}

TOOL_NAMES = [t.name for t in ALL_TOOLS]
