#!/bin/bash

# Glove数据集CPU性能指标测试脚本
# 专门用于验证HNSW搜索过程中的IPC、分支指令和分支预测错误率

# 设置环境变量
source ../set.sh

# 测试数据集 - 专注于glove数据集
datasets=(
    "glove-200-angular"     # 中等维度数据集
)

# 测试参数
K=10
efSearch=100
ef=500
M=16

# 结果目录
result_base="./results/glove_cpu_metrics"
mkdir -p ${result_base}

echo "开始Glove数据集CPU性能指标测试..."
echo "测试指标: IPC, Total Branch Instructions, Branch Misprediction Rate"
echo "=================================================================="

for data in "${datasets[@]}"; do
    echo "正在测试数据集: ${data}"
    
    # 设置数据集特定参数
    if [ "$data" == "glove-200-angular" ]; then
        efSearch=500
        ef=500
    fi
    
    # 路径设置 - 修复路径问题
    data_path=${store_path}/_${data}
    index_path="./DATA/_${data}"  # 使用相对路径
    result_path="${result_base}/${data}"
    temp_data="./DATA/_${data}"
    
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
        
        # 设置索引文件路径 - 修复路径问题
        case $method_id in
            0)
                index="${index_path}/_${data}_ef${ef}_M${M}.index"
                ;;
            1)
                index="${index_path}/O${data}_ef${ef}_M${M}.index"
                ;;
        esac
        
        # 检查索引文件是否存在
        if [ ! -f "$index" ]; then
            echo "    警告: 索引文件不存在: $index"
            echo "    当前工作目录: $(pwd)"
            echo "    尝试列出目录内容:"
            ls -la "${index_path}/" | head -10
            continue
        fi
        
        # 结果文件
        cpu_metrics_log="${result_path}/${data}_${method_name}_cpu_metrics.log"
        perf_log="${result_path}/${data}_${method_name}_perf.log"
        
        echo "    索引文件: $index"
        echo "    结果文件: $cpu_metrics_log"
        
        # 使用perf工具收集CPU性能指标
        echo "    开始性能分析..."
        
        if command -v perf &> /dev/null; then
            echo "    使用perf stat收集CPU指标..."
            
            # 收集关键CPU性能指标
            perf stat -e cycles,instructions,branches,branch-misses,stalled-cycles-frontend,stalled-cycles-backend,cache-misses,cache-references \
                -o ${perf_log} \
                ./cmake-build-debug/src/search_hnsw_dist \
                -d ${method_id} \
                -i ${index} \
                -q ${query} \
                -g ${gnd} \
                -k ${K} \
                -s ${efSearch}
            
            # 解析perf结果并计算指标
            if [ -f "${perf_log}" ]; then
                echo "    解析perf结果..."
                
                # 提取关键指标
                cycles=$(grep "cycles" ${perf_log} | head -1 | awk '{print $1}' | tr -d ',')
                instructions=$(grep "instructions" ${perf_log} | head -1 | awk '{print $1}' | tr -d ',')
                branches=$(grep "branches" ${perf_log} | head -1 | awk '{print $1}' | tr -d ',')
                branch_misses=$(grep "branch-misses" ${perf_log} | head -1 | awk '{print $1}' | tr -d ',')
                stalled_frontend=$(grep "stalled-cycles-frontend" ${perf_log} | head -1 | awk '{print $1}' | tr -d ',')
                stalled_backend=$(grep "stalled-cycles-backend" ${perf_log} | head -1 | awk '{print $1}' | tr -d ',')
                cache_misses=$(grep "cache-misses" ${perf_log} | head -1 | awk '{print $1}' | tr -d ',')
                cache_refs=$(grep "cache-references" ${perf_log} | head -1 | awk '{print $1}' | tr -d ',')
                
                # 计算关键指标
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
                
                if [ -n "$cache_refs" ] && [ -n "$cache_misses" ] && [ "$cache_refs" -gt 0 ]; then
                    cache_miss_rate=$(echo "scale=4; $cache_misses * 100 / $cache_refs" | bc -l 2>/dev/null || echo "N/A")
                else
                    cache_miss_rate="N/A"
                fi
                
                # 保存结果到CPU指标日志
                cat > ${cpu_metrics_log} << EOF
Glove数据集CPU性能指标测试结果
============================
数据集: ${data}
搜索方法: ${method_name}
测试时间: $(date)

关键性能指标:
============
- 总周期数: ${cycles:-N/A}
- 总指令数: ${instructions:-N/A}
- 总分支指令数: ${branches:-N/A}
- 分支预测错误数: ${branch_misses:-N/A}
- IPC (每周期指令数): ${ipc}
- 分支预测错误率: ${misprediction_rate}%
- 前端停滞周期数: ${stalled_frontend:-N/A}
- 后端停滞周期数: ${stalled_backend:-N/A}
- 缓存未命中数: ${cache_misses:-N/A}
- 缓存引用数: ${cache_refs:-N/A}
- 缓存未命中率: ${cache_miss_rate}%

分析说明:
=========
- IPC值越高表示CPU效率越高，理想值接近CPU的理论IPC
- 分支预测错误率越低越好，高错误率会导致CPU停滞
- 前端/后端停滞周期数反映CPU流水线效率
- 缓存未命中率影响内存访问效率

原始perf输出:
=============
$(cat ${perf_log})
EOF
                
                echo "    IPC: ${ipc}"
                echo "    分支指令数: ${branches:-N/A}"
                echo "    分支预测错误率: ${misprediction_rate}%"
                echo "    缓存未命中率: ${cache_miss_rate}%"
            fi
        else
            echo "    警告: perf工具未安装，无法收集CPU指标"
            echo "    请安装perf: sudo apt-get install linux-tools-common linux-tools-generic"
        fi
        
        echo "    完成方法: ${method_name}"
        echo "    ----------------------------------------"
    done
    
    echo "完成数据集: ${data}"
    echo "=================================================================="
done

echo "所有测试完成！"
echo "结果保存在: ${result_base}"

# 生成汇总报告
summary_file="${result_base}/glove_summary_report.txt"
echo "生成汇总报告: ${summary_file}"

cat > ${summary_file} << EOF
Glove数据集CPU性能指标测试汇总报告
================================
测试时间: $(date)
测试数据集: ${datasets[*]}
测试方法: ${search_methods[*]}

结果目录: ${result_base}

测试目的:
========
验证论文中提到的观点：
"后者是由于修剪后的逐向量搜索几乎没有机会并行化工作，因为必须每32个维度评估一次距离，
从而导致CPU停滞的分支预测增加4倍。在剪枝能力低的低维数据集（NYTimes/16、GloVe/50）上，
SIMD-ADS由于无法在每个步骤中充分利用可用的寄存器，因此更加困难"

关键指标说明:
============
1. IPC (Instructions Per Cycle): 
   - 每周期执行的指令数
   - 值越高表示CPU效率越高
   - 理想值接近CPU的理论IPC (如4.0)

2. 分支预测错误率:
   - 分支预测失败的比例
   - 值越低越好
   - 高错误率会导致CPU流水线停滞

3. 分支指令数:
   - 总的分支指令数量
   - 反映代码的复杂度
   - 过多的分支指令可能影响性能

预期结果:
========
- glove-200-angular (中维): 可能有较高的分支预测错误率，IPC可能较高
- 不同搜索方法的性能差异应该能反映SIMD优化的效果

详细结果请查看各数据集目录下的具体日志文件。
EOF

echo "汇总报告已生成: ${summary_file}"
echo ""
echo "使用方法:"
echo "1. 确保已安装perf工具: sudo apt-get install linux-tools-common linux-tools-generic"
echo "2. 运行脚本: ./test_glove_cpu_metrics.sh"
echo "3. 查看结果: ${result_base}" 