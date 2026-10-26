# 需要用户人工介入的清单

执行助手**无法 SSH 进内网服务器**。请通过 GitHub 同步（见 [`COLLAB_GITHUB.md`](COLLAB_GITHUB.md)）。

## 已在开发环境就绪（可随仓库走）

- N5K **元数据 + 官方 split**
- FNDDS **`foods.json`（5624）**
- Schema、骨架、toy/B1/Rule 冒烟、标注指南 v0、`PLAN.md`

## 须在服务器由你完成

1. **开 GitHub 仓库**（建议 private），把本项目 push 上去；把 **仓库 URL** 发给执行助手。  
2. **下载 N5K imagery**（推荐 overhead RGB；全量 tar 可选）— 见 [`SERVER_SETUP.md`](SERVER_SETUP.md)。  
3. **确认 2×4090D** 上 Qwen2.5-VL-7B（或同类）可推理。  
4. W3：**双人试标 30 盘**，kappa ≥ 0.6（[`annotation_guide_v0.md`](annotation_guide_v0.md)）。  
5. W5–7：锁定 **200–500** 金标准排期。

## 禁止误解

- 不要提供旧回归权重。  
- 不要默认开体积工具。  
- 不要把 imagery push 进 GitHub。  
- 元数据就绪 ≠ 全量 RGB 已下载。
