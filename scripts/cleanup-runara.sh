#!/bin/bash
# =============================================================================
# Emergency Cleanup Script - Destroys ALL Runara Instances
# =============================================================================
# Use this if something goes wrong and instances are left running.
# This will destroy ALL droplets tagged with "runara".
# =============================================================================

set -euo pipefail

echo "🚨 EMERGENCY CLEANUP"
echo "===================="
echo ""
echo "This will destroy ALL droplets tagged with 'runara'."
echo ""

# List current runara instances
echo "Current Runara instances:"
doctl compute droplet list --tag-name runara --format ID,Name,PublicIPv4,Status,Memory

DROPLET_IDS=$(doctl compute droplet list --tag-name runara --format ID --no-header)

if [[ -z "$DROPLET_IDS" ]]; then
    echo ""
    echo "✅ No Runara instances found. Nothing to clean up."
    exit 0
fi

echo ""
echo "⚠️  Found $(echo "$DROPLET_IDS" | wc -l) instance(s) to destroy."
echo ""
read -p "Are you sure you want to destroy these instances? (yes/no): " CONFIRM

if [[ "$CONFIRM" == "yes" ]]; then
    echo ""
    echo "Destroying instances..."
    echo "$DROPLET_IDS" | while read -r ID; do
        if [[ -n "$ID" ]]; then
            echo "  Destroying $ID..."
            doctl compute droplet delete "$ID" --force
        fi
    done
    echo ""
    echo "✅ All Runara instances destroyed."
else
    echo ""
    echo "❌ Aborted. No instances destroyed."
fi
