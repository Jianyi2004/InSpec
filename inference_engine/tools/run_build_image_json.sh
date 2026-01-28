#!/bin/bash

# ============================================================================
# 脚本功能：构建多图推理的 JSON 数据集
# 
# 使用方式：
# 方式1（手动模式）：直接编辑 MANUAL_DIRS 数组，然后运行脚本
#   bash run_build_image_json.sh
#
# 方式2（关键字搜索模式）：指定父目录和关键字，自动搜索包含关键字的子孙文件夹
#   bash run_build_image_json.sh --search-dir <父目录> --keyword <关键字> -o <输出文件> [--summary <摘要文件>] [--mode <模式>]
#   例如：bash run_build_image_json.sh --search-dir /data/raw_dataset --keyword "DCDU" -o data/dcdu.json --mode multi
#   
#   支持多个关键字（用逗号或空格分隔，匹配任意一个即可）：
#   bash run_build_image_json.sh --search-dir /data --keyword "BBU安装,防水" -o data/output.json
#   bash run_build_image_json.sh --search-dir /data --keyword "BBU安装 防水" -o data/output.json
#
# 方式3（混合模式）：同时使用手动指定和关键字搜索
#   bash run_build_image_json.sh --search-dir <父目录> --keyword <关键字> -o <输出文件>
#   （会将手动指定的目录和搜索到的目录合并）
# ============================================================================

# 手动指定的目录列表（方式1使用）
MANUAL_DIRS=(
    /data_all/share/datasets/Huawei/HuaweiDefeactDetection/data/raw_dataset/21_室外电源线中频线接地质量检查/合格
)

# 默认参数（方式1使用）
DEFAULT_OUTPUT="data/21-OutdoorLineGrounding/multi.json"
DEFAULT_SUMMARY="data/21-OutdoorLineGrounding/multi.txt"
DEFAULT_MODE="multi"

# ============================================================================
# 解析命令行参数
# ============================================================================
SEARCH_DIR=""
KEYWORD=""
OUTPUT_FILE=""
SUMMARY_FILE=""
MODE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --search-dir)
            SEARCH_DIR="$2"
            shift 2
            ;;
        --keyword)
            KEYWORD="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --summary)
            SUMMARY_FILE="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        -h|--help)
            echo "使用方法："
            echo "  方式1（手动模式）: bash $0"
            echo "  方式2（搜索模式）: bash $0 --search-dir <父目录> --keyword <关键字> -o <输出文件> [--summary <摘要文件>] [--mode <模式>]"
            echo ""
            echo "参数说明："
            echo "  --search-dir    要搜索的父目录"
            echo "  --keyword       文件夹名称中包含的关键字（支持多个，用逗号或空格分隔）"
            echo "                  例如: --keyword \"BBU安装,防水\" 或 --keyword \"BBU安装 防水\""
            echo "  -o, --output    输出 JSON 文件路径"
            echo "  --summary       输出摘要文件路径（可选）"
            echo "  --mode          处理模式：single 或 multi（默认：multi）"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# ============================================================================
# 构建目录列表
# ============================================================================
ROOT_DIR=()

# 如果指定了搜索目录和关键字，则搜索匹配的文件夹
if [[ -n "$SEARCH_DIR" && -n "$KEYWORD" ]]; then
    echo "正在搜索目录: $SEARCH_DIR"
    echo "关键字: $KEYWORD"
    echo "-----------------------------------"
    
    if [[ ! -d "$SEARCH_DIR" ]]; then
        echo "错误: 搜索目录不存在: $SEARCH_DIR"
        exit 1
    fi
    
    # 将关键字字符串分割成数组（支持逗号和空格分隔）
    IFS=', ' read -ra KEYWORDS <<< "$KEYWORD"
    echo "解析到 ${#KEYWORDS[@]} 个关键字: ${KEYWORDS[*]}"
    echo ""
    
    # 使用关联数组去重
    declare -A UNIQUE_DIRS
    
    # 对每个关键字进行搜索
    for kw in "${KEYWORDS[@]}"; do
        # 去除首尾空格
        kw=$(echo "$kw" | xargs)
        if [[ -z "$kw" ]]; then
            continue
        fi
        
        echo "搜索关键字: '$kw'"
        found_count=0
        
        # 使用 find 搜索包含关键字的目录
        while IFS= read -r dir; do
            # 使用关联数组去重
            if [[ -z "${UNIQUE_DIRS[$dir]}" ]]; then
                UNIQUE_DIRS["$dir"]=1
                ROOT_DIR+=("$dir")
                echo "  找到: $dir"
                ((found_count++))
            fi
        done < <(find "$SEARCH_DIR" -type d -name "*${kw}*" 2>/dev/null)
        
        echo "  关键字 '$kw' 找到 $found_count 个目录"
        echo ""
    done
    
    echo "-----------------------------------"
    echo "共找到 ${#ROOT_DIR[@]} 个匹配的目录（已去重）"
    echo ""
fi

# 如果没有使用搜索模式，或者要混合使用，添加手动指定的目录
if [[ -z "$SEARCH_DIR" || -z "$KEYWORD" ]]; then
    # 纯手动模式
    ROOT_DIR=("${MANUAL_DIRS[@]}")
    echo "使用手动指定的 ${#ROOT_DIR[@]} 个目录"
elif [[ ${#MANUAL_DIRS[@]} -gt 0 ]]; then
    # 混合模式：合并手动目录和搜索结果
    echo "合并手动指定的 ${#MANUAL_DIRS[@]} 个目录"
    ROOT_DIR+=("${MANUAL_DIRS[@]}")
    echo "总共 ${#ROOT_DIR[@]} 个目录"
fi

# 检查是否有目录
if [[ ${#ROOT_DIR[@]} -eq 0 ]]; then
    echo "错误: 没有找到任何目录"
    echo "请检查搜索条件或手动指定目录"
    exit 1
fi

# 设置默认值
if [[ -z "$OUTPUT_FILE" ]]; then
    OUTPUT_FILE="$DEFAULT_OUTPUT"
fi

if [[ -z "$MODE" ]]; then
    MODE="$DEFAULT_MODE"
fi

# 构建 Python 命令参数
PYTHON_CMD="python build_multi_image_json.py --root_dir"
for dir in "${ROOT_DIR[@]}"; do
    PYTHON_CMD="$PYTHON_CMD \"$dir\""
done
PYTHON_CMD="$PYTHON_CMD -o \"$OUTPUT_FILE\" --mode \"$MODE\""

if [[ -n "$SUMMARY_FILE" ]]; then
    PYTHON_CMD="$PYTHON_CMD --summary \"$SUMMARY_FILE\""
elif [[ -z "$SEARCH_DIR" || -z "$KEYWORD" ]]; then
    # 手动模式使用默认摘要文件
    PYTHON_CMD="$PYTHON_CMD --summary \"$DEFAULT_SUMMARY\""
fi

# ============================================================================
# 执行 Python 脚本
# ============================================================================
echo ""
echo "执行命令："
echo "$PYTHON_CMD"
echo ""

eval $PYTHON_CMD
