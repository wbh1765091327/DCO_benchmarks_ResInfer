#!/bin/bash

# 清空指定目录下的所有文件，保持目录结构
clear_files() {
    local target_dir="$1"
    
    # 检查目录是否存在
    if [ ! -d "$target_dir" ]; then
        echo "错误: 目录 $target_dir 不存在"
        return 1
    fi
    
    echo "正在清空目录: $target_dir"
    echo "保持目录结构不变..."
    
    # 递归查找所有文件（不包括目录）并删除
    find "$target_dir" -type f -delete
    
    echo "完成！所有文件已删除，目录结构保持不变"
}

# 使用示例
TARGET_DIR="/home/wbh/cppwork/Res-Infer/results/recall@10/"
clear_files "$TARGET_DIR"