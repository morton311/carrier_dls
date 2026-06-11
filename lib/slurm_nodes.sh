#!/bin/bash
# Function to check nodes in use in SLURM
# Usage: check_nodes_in_use [node_prefix] [total_nodes]
# Example: check_nodes_in_use "raider-mla" 32

check_nodes_in_use() {
    local node_prefix="${1:-raider-mla}"
    local total_nodes="${2:-32}"
    
    # Function to expand nodelist notation (e.g., raider-mla[17,29-32] -> raider-mla17 raider-mla29 ... raider-mla32)
    expand_nodelist() {
        local nodelist="$1"
        local prefix="$2"
        
        # If it has bracket notation
        if [[ "$nodelist" == *"["* ]]; then
            # Extract the indices between brackets
            local indices="${nodelist##*\[}"
            indices="${indices%\]}"
            
            # Process each comma-separated item
            echo "$indices" | tr ',' '\n' | while read item; do
                item=$(echo "$item" | xargs)  # trim whitespace
                
                if [[ "$item" == *"-"* ]]; then
                    # It's a range like 29-32
                    local start="${item%-*}"
                    local end="${item#*-}"
                    seq "$start" "$end" | while read num; do
                        echo "${prefix}${num}"
                    done
                else
                    # Single index
                    echo "${prefix}${item}"
                fi
            done
        else
            # Single node without brackets
            echo "$nodelist"
        fi
    }
    
    # Get all unique nodes currently in use
    # Extract the last column (NODELIST) from squeue output, filtering for the specified node prefix
    local unique_nodes=$(squeue -h 2>/dev/null | awk '{print $NF}' | grep -v "None" | grep "^${node_prefix}" | while read nodelist; do
        expand_nodelist "$nodelist" "$node_prefix"
    done | sort -u)
    
    # Count the nodes
    local count=$(echo "$unique_nodes" | grep -c .)
    
    # Display results
    echo "=== SLURM Node Usage ==="
    echo "Nodes in use: $count / $total_nodes"
    echo "Available: $((total_nodes - count))"
}

# Alias for quick access
alias check_nodes='check_nodes_in_use'
