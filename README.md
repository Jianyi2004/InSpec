# InSpec - 工业质检智能系统

基于视觉语言模型和目标检测模型的工业质检解决方案。

## 项目结构

```
InSpec/
├── README.md                    # 本文档
├── inference_engine/            # 推理引擎
│   ├── README.md               # 推理引擎详细文档
│   ├── server/                 # 模型服务
│   ├── tools/                  # 工具集
│   ├── scripts/                # 推理脚本
│   └── prompts/                # Prompt 配置
└── prompt_foundry/             # Prompt 工程工具
```

## 快速开始

### 推理引擎

详见 [inference_engine/README.md](inference_engine/README.md)

```bash
cd inference_engine

# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动服务
cd server
bash vllm_serve.sh

# 3. 运行推理
cd ../scripts
bash run_vllm_client_multi_mode_async.sh
```

## 主要功能

- ✅ **多模态推理** - VLM + 目标检测模型协同
- ✅ **多图推理** - 支持单图/多图组合推理
- ✅ **异步批量处理** - 高性能并发推理
- ✅ **完整工具链** - 数据构建、标注、评估
- ✅ **服务化部署** - 独立的模型服务

## 文档导航

- [推理引擎文档](inference_engine/README.md)
- [模型服务文档](inference_engine/server/SERVER_README.md)
- [工具集文档](inference_engine/tools/TOOLS_README.md)

## 许可证

本项目仅供内部使用。
