#!/bin/bash

# CPU性能指标测试脚本
# 用于验证HNSW搜索过程中的IPC、分支指令和分支预测错误率

# 设置环境变量
source ../set.sh

# 测试数据集
datasets=(
    "glove-200-angular"     # 中等维度数据集
)

# 测试参数
K=10
efSearch=100
ef=500
M=16

# 结果目录
result_base="./results/cpu_metrics"
mkdir -p ${result_base}

echo "开始CPU性能指标测试..."
echo "测试指标: IPC, Total Branch Instructions, Branch Misprediction Rate"
echo "================================================================"

for data in "${datasets[@]}"; do
    echo "正在测试数据集: ${data}"
    
    # 设置数据集特定参数
    if [ "$data" == "glove-200-angular" ]; then
        efSearch=500
        ef=500
    fi
    
    # 路径设置
    data_path=${store_path}/_${data}
    index_path=./DATA/_${data}
    result_path="${result_base}/${data}"
    temp_data=./DATA/_${data}
    
    # 输入文件
    query="${data_path}/_${data}_query.fvecs"
    gnd="${data_path}/_${data}_groundtruth.ivecs"
    
    # 创建结果目录
    mkdir -p ${result_path}
    
    # 测试不同的搜索方法
    search_methods=(
        "0:hnsw_baseline"           # 基础HNSW
        "1:hnsw_adsampling"        # HNSW + ADSampling
    )
    
    for method in "${search_methods[@]}"; do
        IFS=':' read -r method_id method_name <<< "$method"
        
        echo "  测试方法: ${method_name}"
        
        # 设置索引文件路径
        case $method_id in
            0)
                index="${index_path}/_${data}_ef${ef}_M${M}.index"
                trans=""
                code=""
                ;;
            1)
                index="${index_path}/O${data}_ef${ef}_M${M}.index"
                trans="${temp_data}/O.fvecs"
                code=""
                ;;
        esac
         
        # 结果文件
        cpu_metrics_log="${result_path}/${data}_${method_name}_cpu_metrics.log"
        perf_log="${result_path}/${data}_${method_name}_perf.log"
        
        echo "    索引文件: $index"
        echo "    结果文件: $cpu_metrics_log"
        
        # 使用perf工具收集CPU性能指标
        echo "    开始性能分析..."
        
        # 方法1: 使用perf stat收集基本统计信息
        if command -v perf &> /dev/null; then
            echo "    使用perf stat收集CPU指标..."
            
            # 构建完整的命令
            cmd="./cmake-build-debug/src/search_hnsw_dist -d ${method_id} -i ${index} -q ${query} -g ${gnd} -t ${trans} -k ${K} -s ${efSearch}"
            echo "    执行命令: $cmd"
            
            # 收集IPC、分支指令等指标
            perf stat -e cycles,instructions,branches,branch-misses,stalled-cycles-frontend,stalled-cycles-backend \
                -o ${perf_log} \
                ${cmd}
            
            # 解析perf结果并计算指标
            if [ -f "${perf_log}" ]; then
                echo "    解析perf结果..."
                
                # 提取关键指标
                cycles=$(grep "cycles" ${perf_log} | awk '{print $1}' | tr -d ',')
                instructions=$(grep "instructions" ${perf_log} | awk '{print $1}' | tr -d ',')
                branches=$(grep "branches" ${perf_log} | awk '{print $1}' | tr -d ',')
                branch_misses=$(grep "branch-misses" ${perf_log} | awk '{print $1}' | tr -d ',')
                
                # 计算IPC和分支预测错误率
                if [ -n "$cycles" ] && [ -n "$instructions" ] && [ "$cycles" -gt 0 ]; then
                    ipc=$(echo "scale=4; $instructions / $cycles" | bc -l 2>/dev/null || echo "N/A")
                else
                    ipc="N/A"
                fi
                
                if [ -n "$branches" ] && [ -n "$branch_misses" ] && [ "$branches" -gt 0 ]; then
                    misprediction_rate=$(echo "scale=4; $branch_misses * 100 / $branches" | bc -l 2>/dev/null || echo "N/A")
                else
                    misprediction_rate="N/A"
                fi
                
                # 保存结果到CPU指标日志
                cat > ${cpu_metrics_log} << EOF
CPU性能指标测试结果
==================
数据集: ${data}
搜索方法: ${method_name}
测试时间: $(date)

性能指标:
- 总周期数: ${cycles:-N/A}
- 总指令数: ${instructions:-N/A}
- 总分支指令数: ${branches:-N/A}
- 分支预测错误数: ${branch_misses:-N/A}
- IPC (每周期指令数): ${ipc}
- 分支预测错误率: ${misprediction_rate}%

原始perf输出:
$(cat ${perf_log})
EOF
                
                echo "    IPC: ${ipc}"
                echo "    分支指令数: ${branches:-N/A}"
                echo "    分支预测错误率: ${misprediction_rate}%"
            fi
        else
            echo "    警告: perf工具未安装，无法收集CPU指标"
        fi
        
        # 方法2: 使用time命令收集基本时间信息
        echo "    使用time命令收集执行时间..."
        
        # 构建完整的命令
        cmd="./cmake-build-debug/src/search_hnsw_dist -d ${method_id} -i ${index} -q ${query} -g ${gnd} -k ${K} -s ${efSearch}"
        if [ -n "$trans" ]; then
            cmd="${cmd} -t ${trans}"
        fi
        if [ -n "$code" ]; then
            cmd="${cmd} -b ${code}"
        fi
        
        /usr/bin/time -v -o ${result_path}/${data}_${method_name}_time.log ${cmd}
        
        echo "    完成方法: ${method_name}"
        echo "    ----------------------------------------"
    done
    
    echo "完成数据集: ${data}"
    echo "================================================================"
done

echo "所有测试完成！"
echo "结果保存在: ${result_base}"

# 生成汇总报告
summary_file="${result_base}/summary_report.txt"
echo "生成汇总报告: ${summary_file}"

cat > ${summary_file} << EOF
CPU性能指标测试汇总报告
=====================
测试时间: $(date)
测试数据集: ${datasets[*]}
测试方法: ${search_methods[*]}

结果目录: ${result_base}

测试说明:
- IPC (Instructions Per Cycle): 每周期执行的指令数，值越高表示CPU效率越高
- 分支指令数: 总的分支指令数量，反映代码的复杂度
- 分支预测错误率: 分支预测失败的比例，值越低越好

预期结果:
- glove-200-angular (中维): 可能有较高的分支预测错误率，IPC可能较高
- 不同搜索方法的性能差异应该能反映SIMD优化的效果

详细结果请查看各数据集目录下的具体日志文件。
EOF

echo "汇总报告已生成: ${summary_file}" 