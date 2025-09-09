#!/bin/bash
"""
Script目录导航脚本
提供快速访问各个功能模块的便捷方式
"""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 显示标题
show_title() {
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}    Megatron-LM Script 导航工具${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
}

# 显示菜单
show_menu() {
    echo -e "${GREEN}请选择要访问的功能模块:${NC}"
    echo ""
    echo -e "${YELLOW}1.${NC} 数据处理 (data_processing)"
    echo -e "${YELLOW}2.${NC} 可视化工具 (visualization)"
    echo -e "${YELLOW}3.${NC} 工具脚本 (utils)"
    echo -e "${YELLOW}4.${NC} 模板文件 (templates)"
    echo -e "${YELLOW}5.${NC} 训练脚本 (training)"
    echo -e "${YELLOW}6.${NC} 查看目录结构"
    echo -e "${YELLOW}7.${NC} 快速命令"
    echo -e "${YELLOW}8.${NC} 帮助信息"
    echo -e "${YELLOW}0.${NC} 退出"
    echo ""
}

# 显示数据处理菜单
show_data_processing_menu() {
    echo -e "${BLUE}=== 数据处理模块 ===${NC}"
    echo ""
    echo -e "${GREEN}可用脚本:${NC}"
    echo "1. process_dolma_data.sh - 处理Dolma数据集"
    echo "2. process_wikipedia_data.sh - 处理Wikipedia数据集"
    echo "3. process_c4_data.sh - 处理C4数据集"
    echo "4. process_custom_data.sh - 处理自定义数据集"
    echo "5. data_processing_utils.py - 数据处理工具"
    echo "6. 返回主菜单"
    echo ""
    echo -e "${YELLOW}快速命令:${NC}"
    echo "cd data_processing && ls -la"
    echo "cd data_processing && ./process_dolma_data.sh"
    echo "cd data_processing && python data_processing_utils.py --action check"
}

# 显示可视化菜单
show_visualization_menu() {
    echo -e "${BLUE}=== 可视化模块 ===${NC}"
    echo ""
    echo -e "${GREEN}可用脚本:${NC}"
    echo "1. visualize_tensors.py - 完整tensor可视化"
    echo "2. quick_visualize.py - 快速可视化"
    echo "3. one_click_visualize.sh - 一键可视化"
    echo "4. 返回主菜单"
    echo ""
    echo -e "${YELLOW}快速命令:${NC}"
    echo "cd visualization && ls -la"
    echo "cd visualization && ./one_click_visualize.sh"
    echo "cd visualization && python quick_visualize.py"
}

# 显示工具菜单
show_utils_menu() {
    echo -e "${BLUE}=== 工具模块 ===${NC}"
    echo ""
    echo -e "${GREEN}可用脚本:${NC}"
    echo "1. quant_type_modifier.py - 量化类型修改工具"
    echo "2. update_scripts_with_pattern_v2.py - 脚本模式更新工具"
    echo "3. 返回主菜单"
    echo ""
    echo -e "${YELLOW}快速命令:${NC}"
    echo "cd utils && ls -la"
    echo "cd utils && python quant_type_modifier.py --help"
    echo "cd utils && python update_scripts_with_pattern_v2.py --help"
}

# 显示模板菜单
show_templates_menu() {
    echo -e "${BLUE}=== 模板模块 ===${NC}"
    echo ""
    echo -e "${GREEN}可用文件:${NC}"
    echo "1. improved_script_template.sh - 改进的训练脚本模板"
    echo "2. 返回主菜单"
    echo ""
    echo -e "${YELLOW}快速命令:${NC}"
    echo "cd templates && ls -la"
    echo "cd templates && cp improved_script_template.sh ../my_training.sh"
}

# 显示训练脚本菜单
show_training_menu() {
    echo -e "${BLUE}=== 训练脚本模块 ===${NC}"
    echo ""
    echo -e "${GREEN}可用模型:${NC}"
    echo "1. llama32-1b - LLaMA 3.2 1B模型"
    echo "2. llama31-8b - LLaMA 3.1 8B模型"
    echo "3. deepseek2_lite - DeepSeek2 Lite模型"
    echo "4. 返回主菜单"
    echo ""
    echo -e "${YELLOW}快速命令:${NC}"
    echo "cd training && ls -la"
    echo "cd training/llama32-1b && ls -la"
    echo "cd training/llama32-1b && ./pretrain_llama32-1b_dolma_hifp8.sh"
}

# 显示目录结构
show_directory_structure() {
    echo -e "${BLUE}=== 目录结构 ===${NC}"
    echo ""
    echo "script/"
    echo "├── data_processing/          # 数据处理脚本"
    echo "│   ├── process_dolma_data.sh"
    echo "│   ├── process_wikipedia_data.sh"
    echo "│   ├── process_c4_data.sh"
    echo "│   ├── process_custom_data.sh"
    echo "│   ├── data_processing_utils.py"
    echo "│   └── README.md"
    echo "├── visualization/            # 可视化脚本"
    echo "│   ├── visualize_tensors.py"
    echo "│   ├── quick_visualize.py"
    echo "│   ├── one_click_visualize.sh"
    echo "│   └── README.md"
    echo "├── utils/                   # 工具脚本"
    echo "│   ├── quant_type_modifier.py"
    echo "│   ├── update_scripts_with_pattern_v2.py"
    echo "│   └── README.md"
    echo "├── templates/               # 模板文件"
    echo "│   ├── improved_script_template.sh"
    echo "│   └── README.md"
    echo "├── training/                # 训练脚本（按模型分类）"
    echo "│   ├── llama32-1b/"
    echo "│   ├── llama31-8b/"
    echo "│   └── deepseek2_lite/"
    echo "├── navigate.sh              # 本导航脚本"
    echo "└── README.md               # 主说明文档"
    echo ""
}

# 显示快速命令
show_quick_commands() {
    echo -e "${BLUE}=== 快速命令 ===${NC}"
    echo ""
    echo -e "${GREEN}数据处理:${NC}"
    echo "  ./data_processing/process_dolma_data.sh"
    echo "  python data_processing/data_processing_utils.py --action check"
    echo ""
    echo -e "${GREEN}可视化:${NC}"
    echo "  ./visualization/one_click_visualize.sh"
    echo "  python visualization/quick_visualize.py"
    echo ""
    echo -e "${GREEN}工具:${NC}"
    echo "  python utils/quant_type_modifier.py --help"
    echo "  python utils/update_scripts_with_pattern_v2.py"
    echo ""
    echo -e "${GREEN}训练:${NC}"
    echo "  cp templates/improved_script_template.sh my_training.sh"
    echo "  ./training/llama32-1b/pretrain_llama32-1b_dolma_hifp8.sh"
    echo ""
}

# 显示帮助信息
show_help() {
    echo -e "${BLUE}=== 帮助信息 ===${NC}"
    echo ""
    echo -e "${GREEN}使用方法:${NC}"
    echo "1. 运行此脚本: ./navigate.sh"
    echo "2. 选择要访问的功能模块"
    echo "3. 按照提示操作"
    echo ""
    echo -e "${GREEN}各模块功能:${NC}"
    echo "• data_processing: 处理各种数据集，转换为训练格式"
    echo "• visualization: 可视化tensor数据，分析量化效果"
    echo "• utils: 批量修改脚本，管理量化类型"
    echo "• templates: 训练脚本模板，快速创建新脚本"
    echo "• training: 按模型分类的训练脚本"
    echo ""
    echo -e "${GREEN}环境要求:${NC}"
    echo "• Python 3.8+"
    echo "• PyTorch 1.12+"
    echo "• CUDA 11.0+"
    echo "• 必要的Python包: matplotlib, seaborn, pandas, scipy"
    echo ""
    echo -e "${GREEN}环境变量:${NC}"
    echo "export CUSTOM_QUANT_TYPE=\"hifp8\""
    echo "export TENSOR_SAVE_DIR=\"./enhanced_tensor_logs\""
    echo "export TENSOR_SAVE_ENABLED=\"true\""
    echo ""
}

# 处理用户选择
handle_choice() {
    local choice=$1
    
    case $choice in
        1)
            show_data_processing_menu
            echo -e "${YELLOW}按回车键返回主菜单...${NC}"
            read
            ;;
        2)
            show_visualization_menu
            echo -e "${YELLOW}按回车键返回主菜单...${NC}"
            read
            ;;
        3)
            show_utils_menu
            echo -e "${YELLOW}按回车键返回主菜单...${NC}"
            read
            ;;
        4)
            show_templates_menu
            echo -e "${YELLOW}按回车键返回主菜单...${NC}"
            read
            ;;
        5)
            show_training_menu
            echo -e "${YELLOW}按回车键返回主菜单...${NC}"
            read
            ;;
        6)
            show_directory_structure
            echo -e "${YELLOW}按回车键返回主菜单...${NC}"
            read
            ;;
        7)
            show_quick_commands
            echo -e "${YELLOW}按回车键返回主菜单...${NC}"
            read
            ;;
        8)
            show_help
            echo -e "${YELLOW}按回车键返回主菜单...${NC}"
            read
            ;;
        0)
            echo -e "${GREEN}感谢使用！再见！${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}无效选择，请重新输入！${NC}"
            ;;
    esac
}

# 主循环
main() {
    while true; do
        clear
        show_title
        show_menu
        echo -e -n "${GREEN}请输入选择 (0-8): ${NC}"
        read choice
        handle_choice $choice
    done
}

# 检查是否直接运行
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main
fi
