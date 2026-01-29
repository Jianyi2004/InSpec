# Prompt Foundry

专业的工业质检 Prompt 自动生成和优化系统。

## 🚀 快速开始

```bash
# 启动系统
bash start_app.sh

# 访问 Web 界面
http://localhost:7860
```

## 📚 文档

详细文档请查看 `docs/` 目录：

- **[完整文档](docs/README.md)** - 项目完整说明
- **[使用指南](docs/USAGE_GUIDE.md)** - 详细使用教程
- **[功能说明](docs/ENHANCED_FEATURES.md)** - 增强功能详解
- **[完成报告](docs/COMPLETION_REPORT.md)** - 项目完成情况
- **[设计文档](docs/DESIGN.md)** - 系统设计说明
- **[Qwen-Agent 集成](docs/QWEN_AGENT_INTEGRATION.md)** - Agent 框架集成

## ✨ 核心功能

1. **智能生成** - 根据需求自动生成专业 Prompt
2. **智能修改** - 对现有 Prompt 进行精准修改
3. **版本管理** - 完整的版本历史和回溯
4. **差异对比** - 类似 git diff 的可视化对比
5. **示例管理** - ICL 示例图片管理
6. **图片检测** - 自动检测缺失的示例图片

## 📁 项目结构

```
prompt_foundry/
├── README.md              # 项目入口文档
├── start_app.sh           # 启动脚本
├── requirements.txt       # Python 依赖
├── core/                  # 核心模块
│   ├── version_manager.py # 版本管理
│   ├── diff_generator.py  # 差异对比
│   ├── qwen_tools.py      # Qwen-Agent 工具
│   └── qwen_agents.py     # Qwen-Agent 代理
├── web/                   # Web 界面
│   └── app_enhanced.py    # 增强版界面
├── prompts/               # Prompt 存储
│   └── {任务名}/
│       ├── prompt.txt     # 当前版本
│       ├── icl/           # 示例图片
│       └── versions/      # 版本历史
├── scripts/               # 工具脚本
└── docs/                  # 文档目录
```

## 🔧 配置

编辑 `start_app.sh` 配置 vLLM 服务：

```bash
SERVER_URL="http://10.249.42.141:7878/v1"
MODEL="/data_all/share/models/Qwen3-VL-32B-Instruct"
```

## 📝 版本管理

每个版本包含以下文件：

```
versions/v{major}.{minor}_{tag}_{timestamp}/
├── prompt.txt             # 完整 prompt
├── system_prompt.txt      # 系统提示词
├── main_prompt.txt        # 主提示词
├── summary_prompt.txt     # 总结提示词
├── icl/                   # 示例图片
└── version_info.json      # 版本信息
```

## 🎯 使用场景

- 生成新的质检 Prompt
- 修改现有 Prompt 规则
- 管理 Prompt 版本历史
- 对比不同版本差异
- 管理 ICL 示例图片

## 📞 技术支持

查看详细文档：`docs/README.md`
