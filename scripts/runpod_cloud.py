#!/usr/bin/env python3
"""RunPod GPU pod management.

Usage:
    python scripts/runpod_cloud.py types                           # list GPU types + pricing
    python scripts/runpod_cloud.py status                          # list pods
    python scripts/runpod_cloud.py launch [--gpu TYPE] [--image IMG]
    python scripts/runpod_cloud.py ssh [pod_id]                    # interactive SSH
    python scripts/runpod_cloud.py run <command>                   # run command via SSH
    python scripts/runpod_cloud.py stop [pod_id]                   # stop (preserves volume)
    python scripts/runpod_cloud.py start [pod_id]                  # resume stopped pod
    python scripts/runpod_cloud.py terminate [pod_id] [--yes]      # permanent delete
    python scripts/runpod_cloud.py setup [pod_id]                  # install Claude Code + deps

Requires RUNPOD_API_KEY environment variable (rpa_... key from RunPod console).
"""

import argparse
import json
import os
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request

try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = ssl.create_default_context()
    _SSL_CTX.load_default_certs()
    if not _SSL_CTX.get_ca_certs():
        _SSL_CTX = ssl._create_unverified_context()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REST_BASE = "https://rest.runpod.io/v1"
GRAPHQL_URL = "https://api.runpod.io/graphql"

DEFAULT_GPU = "NVIDIA A100 80GB PCIe"
DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
DEFAULT_VOLUME_GB = 150
DEFAULT_CONTAINER_DISK_GB = 150
DEFAULT_PORTS = ["8888/http", "22/tcp"]
DEFAULT_CLOUD_TYPE = "COMMUNITY"

SSH_KEY_PATH = os.path.expanduser("~/.ssh/id_ed25519")
SSH_USER = "root"

# Display name → API gpuTypeId mapping.  The `types` command shows short display
# names but the launch API requires the full enum value.
_GPU_DISPLAY_TO_API = {
    "RTX 3070":             "NVIDIA GeForce RTX 3070",
    "RTX 3080":             "NVIDIA GeForce RTX 3080",
    "RTX 3080 Ti":          "NVIDIA GeForce RTX 3080 Ti",
    "RTX 3090":             "NVIDIA GeForce RTX 3090",
    "RTX 3090 Ti":          "NVIDIA GeForce RTX 3090 Ti",
    "RTX 4070 Ti":          "NVIDIA GeForce RTX 4070 Ti",
    "RTX 4080":             "NVIDIA GeForce RTX 4080",
    "RTX 4080 SUPER":       "NVIDIA GeForce RTX 4080 SUPER",
    "RTX 4090":             "NVIDIA GeForce RTX 4090",
    "RTX 5080":             "NVIDIA GeForce RTX 5080",
    "RTX 5090":             "NVIDIA GeForce RTX 5090",
    "RTX A4000":            "NVIDIA RTX A4000",
    "RTX A4500":            "NVIDIA RTX A4500",
    "RTX A5000":            "NVIDIA RTX A5000",
    "RTX A6000":            "NVIDIA RTX A6000",
    "RTX 4000 Ada":         "NVIDIA RTX 4000 Ada Generation",
    "RTX 5000 Ada":         "NVIDIA RTX 5000 Ada Generation",
    "RTX 6000 Ada":         "NVIDIA RTX 6000 Ada Generation",
    "RTX 2000 Ada":         "NVIDIA RTX 2000 Ada Generation",
    "RTX PRO 4500":         "NVIDIA RTX PRO 4500",
    "RTX PRO 6000":         "NVIDIA RTX PRO 6000 Blackwell Server Edition",
    "RTX PRO 6000 WK":      "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
    "RTX PRO 6000 MaxQ":    "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
    "A30":                  "NVIDIA A30",
    "A40":                  "NVIDIA A40",
    "A100 PCIe":            "NVIDIA A100 80GB PCIe",
    "A100 SXM":             "NVIDIA A100-SXM4-80GB",
    "L4":                   "NVIDIA L4",
    "L40":                  "NVIDIA L40",
    "L40S":                 "NVIDIA L40S",
    "H100 PCIe":            "NVIDIA H100 PCIe",
    "H100 NVL":             "NVIDIA H100 NVL",
    "H100 SXM":             "NVIDIA H100 80GB HBM3",
    "H200 SXM":             "NVIDIA H200",
    "H200 NVL":             "NVIDIA H200 NVL",
    "B200":                 "NVIDIA B200",
    "MI300X":               "AMD Instinct MI300X OAM",
    "Tesla V100":           "Tesla V100-PCIE-16GB",
    "V100 SXM2":            "Tesla V100-SXM2-16GB",
    "V100 SXM2 32GB":       "Tesla V100-SXM2-32GB",
}


def _resolve_gpu_id(user_input: str) -> str:
    """Map a short display name to the API gpuTypeId.

    Accepts the full API ID, a short display name from _GPU_DISPLAY_TO_API,
    or a case-insensitive substring match.
    """
    # Exact match on API ID
    if user_input in _GPU_DISPLAY_TO_API.values():
        return user_input

    # Exact match on short name
    if user_input in _GPU_DISPLAY_TO_API:
        return _GPU_DISPLAY_TO_API[user_input]

    # Case-insensitive match on short name
    lower = user_input.lower()
    for short, full in _GPU_DISPLAY_TO_API.items():
        if short.lower() == lower:
            return full

    # Substring match (case-insensitive) on both short and full names
    matches = []
    for short, full in _GPU_DISPLAY_TO_API.items():
        if lower in short.lower() or lower in full.lower():
            matches.append((short, full))

    if len(matches) == 1:
        return matches[0][1]
    if len(matches) > 1:
        print(f"Ambiguous GPU name '{user_input}'. Matches:", file=sys.stderr)
        for short, full in matches:
            print(f"  {short:<25} → {full}", file=sys.stderr)
        sys.exit(1)

    # No match — pass through and let the API reject it
    return user_input

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def _get_api_key():
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        print("Error: RUNPOD_API_KEY environment variable not set.", file=sys.stderr)
        print("Get your key from RunPod Console > Settings > API Keys", file=sys.stderr)
        sys.exit(1)
    return key


def _rest(method, path, data=None):
    """Make an authenticated request to the RunPod REST API."""
    url = f"{REST_BASE}{path}"
    key = _get_api_key()

    headers = {
        "Authorization": f"Bearer {key}",
        "User-Agent": "runpod-cli/1.0",
        "Accept": "application/json",
    }
    body = None
    if data is not None:
        body = json.dumps(data).encode()
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, context=_SSL_CTX) as resp:
            raw = resp.read().decode()
            if not raw:
                return {}
            return json.loads(raw)
    except urllib.error.HTTPError as e:
        err_body = e.read().decode()
        try:
            err = json.loads(err_body)
            msg = err.get("error", err.get("message", err_body))
        except (json.JSONDecodeError, AttributeError):
            msg = err_body
        print(f"API error ({e.code}): {msg}", file=sys.stderr)
        sys.exit(1)


def _graphql(query, variables=None):
    """Make a GraphQL request to the RunPod API."""
    key = _get_api_key()
    url = f"{GRAPHQL_URL}?api_key={key}"

    payload = {"query": query}
    if variables:
        payload["variables"] = variables

    body = json.dumps(payload).encode()
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "runpod-cli/1.0",
    }

    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, context=_SSL_CTX) as resp:
            result = json.loads(resp.read().decode())
            if "errors" in result:
                print(f"GraphQL error: {result['errors']}", file=sys.stderr)
                sys.exit(1)
            return result.get("data", {})
    except urllib.error.HTTPError as e:
        err_body = e.read().decode()
        print(f"GraphQL error ({e.code}): {err_body}", file=sys.stderr)
        sys.exit(1)


def _get_pods():
    """Return list of pod dicts."""
    return _rest("GET", "/pods")


def _get_pod(pod_id):
    """Return a single pod dict."""
    return _rest("GET", f"/pods/{pod_id}")


def _pick_pod(pods, pod_id=None):
    """Select a pod by id, or auto-select if only one exists."""
    if isinstance(pods, dict):
        pods = [pods]
    running = [p for p in pods if p.get("desiredStatus") == "RUNNING"]
    all_pods = pods

    if pod_id:
        for p in all_pods:
            if p["id"] == pod_id or p["id"].startswith(pod_id):
                return p
        print(f"Error: no pod matching '{pod_id}'", file=sys.stderr)
        sys.exit(1)

    # Auto-select: prefer running, fall back to any
    candidates = running if running else all_pods
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 0:
        print("Error: no pods found.", file=sys.stderr)
        sys.exit(1)

    print("Multiple pods found — specify a pod ID:", file=sys.stderr)
    for p in candidates:
        gpu = p.get("gpu", {}).get("displayName", "?") if isinstance(p.get("gpu"), dict) else "?"
        print(f"  {p['id']}  {p.get('name', '?'):<20} {gpu:<30} {p.get('desiredStatus', '?')}",
              file=sys.stderr)
    sys.exit(1)


def _get_ssh_info(pod):
    """Extract SSH host and port from a pod dict."""
    ip = pod.get("publicIp")
    port_mappings = pod.get("portMappings")

    ssh_port = None
    if port_mappings and isinstance(port_mappings, dict):
        ssh_port = port_mappings.get("22")
    elif port_mappings and isinstance(port_mappings, str):
        # Sometimes returned as "22/tcp -> 0.0.0.0:10341" format
        try:
            mapping = json.loads(port_mappings)
            ssh_port = mapping.get("22")
        except (json.JSONDecodeError, TypeError):
            pass

    if not ip or not ssh_port:
        # Try runtime ports from the pod
        runtime = pod.get("runtime", {})
        if runtime:
            ports = runtime.get("ports", [])
            for p in ports:
                if isinstance(p, dict) and p.get("privatePort") == 22:
                    ip = ip or p.get("ip")
                    ssh_port = ssh_port or p.get("publicPort")

    return ip, ssh_port


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_types(args):
    """List available GPU types with pricing."""
    query = """
    {
        gpuTypes {
            id
            displayName
            memoryInGb
            secureCloud
            communityCloud
            lowestPrice(input: {gpuCount: 1}) {
                uninterruptablePrice
                minimumBidPrice
                stockStatus
            }
        }
    }
    """
    data = _graphql(query)
    gpu_types = data.get("gpuTypes", [])

    # Filter by cloud type
    if args.secure:
        gpu_types = [g for g in gpu_types if g.get("secureCloud")]
    elif args.community:
        gpu_types = [g for g in gpu_types if g.get("communityCloud")]

    # Sort by price (on-demand), nulls last
    def sort_key(g):
        lp = g.get("lowestPrice", {}) or {}
        price = lp.get("uninterruptablePrice")
        return price if price is not None else 9999

    gpu_types.sort(key=sort_key)

    cloud_label = " (secure)" if args.secure else " (community)" if args.community else ""
    print(f"{'GPU Type':<40} {'VRAM':>6} {'On-Demand':>10} {'Spot':>8} {'Cloud':>5} {'Stock'}")
    print("-" * 100)
    for g in gpu_types:
        name = g.get("displayName") or g.get("id", "?")
        mem = g.get("memoryInGb", "?")
        lp = g.get("lowestPrice", {}) or {}
        on_demand = lp.get("uninterruptablePrice")
        spot = lp.get("minimumBidPrice")
        stock = lp.get("stockStatus", "?")

        od_str = f"${on_demand:.2f}/hr" if on_demand else "N/A"
        spot_str = f"${spot:.2f}" if spot else "—"
        mem_str = f"{mem}GB" if mem else "?"
        cloud = "S" if g.get("secureCloud") else ""
        cloud += "C" if g.get("communityCloud") else ""

        print(f"{name:<40} {mem_str:>6} {od_str:>10} {spot_str:>8} {cloud:>5} {stock}")


def cmd_status(args):
    """List all pods."""
    pods = _get_pods()
    if not pods:
        print("No pods.")
        return

    print(f"{'ID':<28} {'Name':<20} {'GPU':<30} {'Status':<12} {'SSH'}")
    print("-" * 110)
    for p in pods:
        gpu = "?"
        if isinstance(p.get("gpu"), dict):
            gpu = p["gpu"].get("displayName", "?")
        elif isinstance(p.get("gpu"), str):
            gpu = p["gpu"]

        ip, port = _get_ssh_info(p)
        ssh_str = f"{ip}:{port}" if ip and port else "—"

        print(f"{p['id']:<28} {p.get('name', '?'):<20} {gpu:<30} "
              f"{p.get('desiredStatus', '?'):<12} {ssh_str}")


def cmd_launch(args):
    """Create a new pod and wait until it's running."""
    gpu_id = _resolve_gpu_id(args.gpu)
    cloud_type = "SECURE" if args.secure else args.cloud_type
    payload = {
        "name": args.name,
        "imageName": args.image,
        "gpuTypeIds": [gpu_id],
        "gpuCount": args.gpu_count,
        "cloudType": cloud_type,
        "containerDiskInGb": args.container_disk,
        "volumeInGb": args.volume,
        "volumeMountPath": "/workspace",
        "ports": DEFAULT_PORTS,
        "supportPublicIp": True,
    }

    if args.spot:
        payload["interruptible"] = True

    print(f"Launching {gpu_id} ({cloud_type})...")
    print(f"  Image: {args.image}")
    print(f"  Volume: {args.volume}GB, Container disk: {args.container_disk}GB")

    resp = _rest("POST", "/pods", payload)
    pod_id = resp.get("id")
    if not pod_id:
        print(f"Error: launch returned no pod ID.", file=sys.stderr)
        print(f"Response: {json.dumps(resp, indent=2)}", file=sys.stderr)
        sys.exit(1)

    print(f"Pod ID: {pod_id}")
    print("Waiting for pod to start...")

    start = time.time()
    max_wait = 300
    interval = 5
    while True:
        elapsed = time.time() - start
        if elapsed > max_wait:
            print(f"\nTimeout after {max_wait}s. Check status manually:", file=sys.stderr)
            print(f"  python scripts/runpod_cloud.py status", file=sys.stderr)
            sys.exit(1)

        pod = _get_pod(pod_id)
        status = pod.get("desiredStatus", "unknown")
        ip, port = _get_ssh_info(pod)

        if status == "RUNNING" and ip and port:
            print(f"\nPod running! SSH: {SSH_USER}@{ip} -p {port}")
            print(f"\nSSH command:")
            print(f"  ssh -i {SSH_KEY_PATH} -p {port} "
                  f"-o StrictHostKeyChecking=no {SSH_USER}@{ip}")
            print(f"\nOr use:")
            print(f"  python scripts/runpod_cloud.py ssh")
            return

        mins = int(elapsed) // 60
        secs = int(elapsed) % 60
        print(f"  [{mins}:{secs:02d}] status: {status}"
              f"{f'  ip: {ip}' if ip else ''}", end="\r")
        time.sleep(interval)


def cmd_ssh(args):
    """Open interactive SSH session to a pod."""
    pods = _get_pods()
    pod = _pick_pod(pods, _resolve_pod_id(args))
    ip, port = _get_ssh_info(pod)

    if not ip or not port:
        print(f"Error: pod {pod['id']} has no SSH endpoint "
              f"(status: {pod.get('desiredStatus')})", file=sys.stderr)
        sys.exit(1)

    ssh_args = [
        "ssh",
        "-i", SSH_KEY_PATH,
        "-p", str(port),
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        f"{SSH_USER}@{ip}",
    ]
    print(f"Connecting to {ip}:{port}...")
    os.execvp("ssh", ssh_args)


def cmd_run(args):
    """Run a command on a pod via SSH."""
    if not args.remote_cmd:
        print("Error: no command specified.", file=sys.stderr)
        sys.exit(1)

    pods = _get_pods()
    pod = _pick_pod(pods, _resolve_pod_id(args))
    ip, port = _get_ssh_info(pod)

    if not ip or not port:
        print(f"Error: pod {pod['id']} has no SSH endpoint "
              f"(status: {pod.get('desiredStatus')})", file=sys.stderr)
        sys.exit(1)

    command = " ".join(args.remote_cmd)
    ssh_args = [
        "ssh",
        "-i", SSH_KEY_PATH,
        "-p", str(port),
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        f"{SSH_USER}@{ip}",
        command,
    ]
    print(f"Running on {ip}:{port}: {command}")
    result = subprocess.run(ssh_args)
    sys.exit(result.returncode)


def cmd_stop(args):
    """Stop a pod (preserves volume, stops GPU billing)."""
    pods = _get_pods()
    pod = _pick_pod(pods, _resolve_pod_id(args))

    print(f"Stopping pod {pod['id']} ({pod.get('name', '?')})...")
    _rest("POST", f"/pods/{pod['id']}/stop")
    print("Done. Volume preserved. Use 'start' to resume.")


def cmd_start(args):
    """Resume a stopped pod."""
    pods = _get_pods()
    pod = _pick_pod(pods, _resolve_pod_id(args))

    print(f"Starting pod {pod['id']} ({pod.get('name', '?')})...")
    _rest("POST", f"/pods/{pod['id']}/start")
    print("Starting... Use 'status' to check when ready.")


def cmd_setup(args):
    """Install Claude Code, Python deps, and configure the pod for IMR work.

    Runs setup_and_delete.py first (if --skip-base is not set) to clone the
    repo and install base deps, then installs Claude Code and writes a
    CLAUDE.md with the project context.
    """
    pods = _get_pods()
    pod = _pick_pod(pods, _resolve_pod_id(args))
    ip, port = _get_ssh_info(pod)

    if not ip or not port:
        print(f"Error: pod {pod['id']} has no SSH endpoint "
              f"(status: {pod.get('desiredStatus')})", file=sys.stderr)
        sys.exit(1)

    ssh_base = [
        "ssh",
        "-i", SSH_KEY_PATH,
        "-p", str(port),
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        f"{SSH_USER}@{ip}",
    ]

    def run_remote(cmd, check=True):
        print(f"  $ {cmd}")
        result = subprocess.run(ssh_base + [cmd])
        if check and result.returncode != 0:
            print(f"Error: command failed (exit {result.returncode})", file=sys.stderr)
            sys.exit(1)
        return result

    # Step 1: Base setup (repo clone, deps) via setup_and_delete.py
    if not args.skip_base:
        print("=== Step 1: Base setup (repo + deps) ===")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        result = subprocess.run(
            [sys.executable, os.path.join(script_dir, "setup_and_delete.py")]
            + (["--pod-id", _resolve_pod_id(args)] if _resolve_pod_id(args) else []),
        )
        if result.returncode != 0:
            print("Base setup failed.", file=sys.stderr)
            sys.exit(1)
        print()

    # Step 2: Install Claude Code (native installer, no Node.js needed)
    print("=== Step 2: Install Claude Code ===")
    run_remote("command -v claude || curl -fsSL https://claude.ai/install.sh | bash")

    # Step 3: Write CLAUDE.md for the submissions project
    print("\n=== Step 3: Configure project CLAUDE.md ===")
    claude_md = r"""# CLAUDE.md — valid-json-wrong-answer (RunPod)

## Project
Per-grammar-role loss decomposition for structured JSON output evaluation.

## Environment (RunPod)
```bash
export HF_HOME=/workspace/hf_cache
cd /workspace/valid-json-wrong-answer
```

## Key Commands
```bash
# Smoke test
bash scripts/smoketest.sh

# Run experiments (32B first, then 7B, then 0.5B)
bash scripts/run_experiment.sh all

# Or run one scale at a time
bash scripts/run_experiment.sh 32b
bash scripts/run_experiment.sh 7b
bash scripts/run_experiment.sh 05b

# Build all paper tables (LaTeX bodies + console preview)
python scripts/build_tables.py
```

## Conventions
- Infrastructure code: strict invariants, raise on violations, no defensive coding
- Standard LoRA via PEFT — no custom adapters
- Default models: Qwen 2.5 Instruct at 0.5B, 7B, 32B
"""

    # Write via heredoc over SSH
    run_remote(
        "cat > /workspace/valid-json-wrong-answer/CLAUDE.md << 'CLAUDEMD_EOF'\n"
        + claude_md
        + "CLAUDEMD_EOF"
    )

    # Step 4: Print next steps
    print("\n=== Setup complete ===")
    print(f"\nSSH into the pod:")
    print(f"  python scripts/runpod_cloud.py ssh")


def cmd_resize(args):
    """Resize volume and/or container disk on a pod.

    Pod must be stopped first. Disk can only be increased, not decreased.
    """
    pods = _get_pods()
    pod = _pick_pod(pods, _resolve_pod_id(args))

    if pod.get("desiredStatus") == "RUNNING":
        print(f"Pod {pod['id']} is running. Stop it first:")
        print(f"  python scripts/runpod_cloud.py stop {pod['id']}")
        sys.exit(1)

    update = {}
    if args.volume:
        update["volumeInGb"] = args.volume
    if args.container_disk:
        update["containerDiskInGb"] = args.container_disk

    if not update:
        print("Error: specify --volume and/or --container-disk", file=sys.stderr)
        sys.exit(1)

    print(f"Resizing pod {pod['id']} ({pod.get('name', '?')})...")
    for k, v in update.items():
        print(f"  {k}: {v}GB")

    _rest("PATCH", f"/pods/{pod['id']}", update)
    print("Done. Start the pod to use new disk sizes:")
    print(f"  python scripts/runpod_cloud.py start {pod['id']}")


def cmd_terminate(args):
    """Permanently delete a pod and all its data."""
    pods = _get_pods()
    pod = _pick_pod(pods, _resolve_pod_id(args))

    gpu = "?"
    if isinstance(pod.get("gpu"), dict):
        gpu = pod["gpu"].get("displayName", "?")

    if not args.yes:
        print(f"PERMANENTLY DELETE pod {pod['id'][:12]}... "
              f"({pod.get('name', '?')}, {gpu})?")
        print("This will destroy all data including persistent volumes.")
        answer = input("Type 'yes' to confirm: ")
        if answer.strip().lower() != "yes":
            print("Aborted.")
            return

    print(f"Terminating {pod['id']}...")
    _rest("DELETE", f"/pods/{pod['id']}")
    print("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_pod_id_arg(parser):
    """Add both positional and --pod-id flag to a subparser."""
    parser.add_argument("pod_id", nargs="?", default=None,
                        help="Pod ID (auto-selects if only one)")
    parser.add_argument("--pod-id", dest="pod_id_flag", default=None,
                        help="Pod ID (alternative to positional)")


def _resolve_pod_id(args):
    """Return pod_id from either positional or --pod-id flag."""
    return args.pod_id or getattr(args, "pod_id_flag", None)


def main():
    parser = argparse.ArgumentParser(
        description="RunPod GPU pod management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # types
    p_types = sub.add_parser("types", help="List available GPU types and pricing")
    p_types.add_argument("--secure", action="store_true", help="Show only secure cloud GPUs")
    p_types.add_argument("--community", action="store_true", help="Show only community cloud GPUs")

    # status
    sub.add_parser("status", help="List all pods")

    # launch
    p_launch = sub.add_parser("launch", help="Launch a new pod")
    p_launch.add_argument("--gpu", default=DEFAULT_GPU,
                          help=f"GPU type (default: {DEFAULT_GPU})")
    p_launch.add_argument("--image", default=DEFAULT_IMAGE,
                          help=f"Docker image (default: {DEFAULT_IMAGE})")
    p_launch.add_argument("--name", default="training",
                          help="Pod name (default: training)")
    p_launch.add_argument("--gpu-count", type=int, default=1,
                          help="Number of GPUs (default: 1)")
    p_launch.add_argument("--volume", type=int, default=DEFAULT_VOLUME_GB,
                          help=f"Persistent volume GB (default: {DEFAULT_VOLUME_GB})")
    p_launch.add_argument("--container-disk", type=int, default=DEFAULT_CONTAINER_DISK_GB,
                          help=f"Container disk GB (default: {DEFAULT_CONTAINER_DISK_GB})")
    p_launch.add_argument("--cloud-type", default=DEFAULT_CLOUD_TYPE,
                          choices=["SECURE", "COMMUNITY", "ALL"],
                          help="Cloud type (default: COMMUNITY)")
    p_launch.add_argument("--secure", action="store_true",
                          help="Shorthand for --cloud-type SECURE")
    p_launch.add_argument("--spot", action="store_true",
                          help="Use spot/interruptible pricing")

    # ssh
    p_ssh = sub.add_parser("ssh", help="SSH into a pod")
    _add_pod_id_arg(p_ssh)

    # run
    p_run = sub.add_parser("run", help="Run a command on a pod via SSH")
    _add_pod_id_arg(p_run)
    p_run.add_argument("remote_cmd", nargs=argparse.REMAINDER, help="Command to run")

    # stop
    p_stop = sub.add_parser("stop", help="Stop a pod (preserves volume)")
    _add_pod_id_arg(p_stop)

    # start
    p_start = sub.add_parser("start", help="Resume a stopped pod")
    _add_pod_id_arg(p_start)

    # resize
    p_resize = sub.add_parser("resize", help="Resize volume/container disk (pod must be stopped)")
    _add_pod_id_arg(p_resize)
    p_resize.add_argument("--volume", type=int, default=None,
                          help="New persistent volume size in GB")
    p_resize.add_argument("--container-disk", type=int, default=None,
                          help="New container disk size in GB")

    # terminate
    p_term = sub.add_parser("terminate", help="Permanently delete a pod")
    _add_pod_id_arg(p_term)
    p_term.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")

    # setup
    p_setup = sub.add_parser("setup", help="Install Claude Code + deps on a pod")
    _add_pod_id_arg(p_setup)
    p_setup.add_argument("--skip-base", action="store_true",
                         help="Skip base setup (repo clone, deps) — only install Claude Code")

    args = parser.parse_args()
    {
        "types": cmd_types,
        "status": cmd_status,
        "launch": cmd_launch,
        "ssh": cmd_ssh,
        "run": cmd_run,
        "stop": cmd_stop,
        "start": cmd_start,
        "resize": cmd_resize,
        "terminate": cmd_terminate,
        "setup": cmd_setup,
    }[args.command](args)


if __name__ == "__main__":
    main()
