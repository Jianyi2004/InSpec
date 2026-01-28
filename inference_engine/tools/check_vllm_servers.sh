#!/bin/bash

# VLLM服务器状态检查脚本
# 显示端口、模型路径、GPU使用和连接状态

echo "=========================================="
echo "VLLM Server 状态汇总"
echo "=========================================="
echo ""

# 获取所有vllm serve进程
vllm_processes=$(ps aux | grep "vllm serve" | grep -v grep)

if [ -z "$vllm_processes" ]; then
    echo "未发现运行中的VLLM服务器"
    exit 0
fi

# 表头
printf "%-6s %-18s %-45s %-15s %-8s %-10s\n" "端口" "用户" "模型路径" "GPU" "TP大小" "连接数"
echo "--------------------------------------------------------------------------------------------------------"

# 解析每个vllm serve进程
echo "$vllm_processes" | while IFS= read -r line; do
    # 提取用户名
    user=$(echo "$line" | awk '{print $1}')
    
    # 提取PID
    pid=$(echo "$line" | awk '{print $2}')
    
    # 提取端口号
    port=$(echo "$line" | grep -oP '(?<=--port )\d+')
    
    # 提取模型路径
    model=$(echo "$line" | grep -oP '(?<=vllm serve )[^ ]+')
    
    # 提取tensor-parallel-size (如果有)
    tp_size=$(echo "$line" | grep -oP '(?<=--tensor-parallel-size )\d+')
    if [ -z "$tp_size" ]; then
        tp_size="1"
    fi
    
    # 获取该端口的活跃连接数
    connections=$(ss -tn 2>/dev/null | grep ":$port " | wc -l)
    
    # 查找使用的GPU (通过nvidia-smi查找该用户的VLLM进程)
    gpu_info=$(nvidia-smi --query-compute-apps=pid,gpu_name,used_memory --format=csv,noheader 2>/dev/null | grep -E "VLLM" | head -n $tp_size)
    
    # 提取GPU编号
    gpu_ids=""
    if [ ! -z "$gpu_info" ]; then
        # 从nvidia-smi获取GPU ID
        gpu_ids=$(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader 2>/dev/null | \
                  grep -E "VLLM" | \
                  head -n $tp_size | \
                  while read -r gpu_line; do
                      gpu_pid=$(echo "$gpu_line" | cut -d',' -f1 | tr -d ' ')
                      # 通过PID找GPU ID
                      nvidia-smi --query-compute-apps=pid,gpu_bus_id --format=csv,noheader 2>/dev/null | \
                      grep "^$gpu_pid" | head -1
                  done)
        
        # 简化：直接从gpustat风格输出获取
        gpu_nums=$(ps aux | grep -E "VLLM.*(Worker|Engine)" | grep "$user" | head -n $tp_size | while read -r proc; do
            proc_pid=$(echo "$proc" | awk '{print $2}')
            nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -n "^$proc_pid$" | cut -d: -f1
        done | tr '\n' ',' | sed 's/,$//')
        
        if [ -z "$gpu_nums" ]; then
            gpu_nums="N/A"
        fi
    else
        gpu_nums="N/A"
    fi
    
    # 简化模型路径显示
    model_short=$(basename "$model")
    
    # 格式化输出
    printf "%-6s %-18s %-45s %-15s %-8s %-10s\n" \
        "$port" "$user" "$model_short" "$gpu_nums" "$tp_size" "$connections"
done

echo ""
echo "=========================================="
echo "详细信息："
echo "=========================================="

# 显示每个端口的详细连接状态
echo "$vllm_processes" | while IFS= read -r line; do
    port=$(echo "$line" | grep -oP '(?<=--port )\d+')
    user=$(echo "$line" | awk '{print $1}')
    model=$(echo "$line" | grep -oP '(?<=vllm serve )[^ ]+')
    
    echo ""
    echo "端口 $port ($user):"
    echo "  模型: $model"
    
    # 显示GPU内存使用
    echo -n "  GPU内存: "
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | \
        grep -E "VLLM" | \
        awk '{sum+=$2} END {if(NR>0) printf "%.1f GB\n", sum/1024; else print "N/A"}'
    
    # 显示连接状态
    active_conns=$(ss -tn 2>/dev/null | grep ":$port " | grep ESTAB | wc -l)
    total_conns=$(ss -tn 2>/dev/null | grep ":$port " | wc -l)
    echo "  连接状态: $active_conns 活跃 / $total_conns 总计"
    
    # 显示进程运行时间
    pid=$(echo "$line" | awk '{print $2}')
    runtime=$(ps -p $pid -o etime= 2>/dev/null | tr -d ' ')
    if [ ! -z "$runtime" ]; then
        echo "  运行时长: $runtime"
    fi
done

echo ""
echo "=========================================="
echo "提示: 使用 'ss -tn | grep :<端口>' 查看具体连接详情"
echo "=========================================="
