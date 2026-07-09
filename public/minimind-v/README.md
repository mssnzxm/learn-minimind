# MiniMind-V 学习教程站

这是一个面向 MiniMind-V 的中文静态教程网站，内容覆盖基础概念、模型架构、数据流程、训练复现、推理部署、代码导览、实验评估和资料索引。

## 本地预览

```bash
python3 -m http.server 8000
```

然后访问：

```text
http://127.0.0.1:8000/
```

也可以直接打开 `index.html`，但某些受限浏览器环境可能会拦截 `file://`，推荐使用本地 HTTP 服务预览。

## Cloudflare Pages 部署

- Framework preset: `None`
- Build command: 留空
- Build output directory: `/`
- Root directory: 仓库根目录

所有页面和静态资源都已经放在仓库内：

```text
.
├── index.html
├── foundations.html
├── architecture.html
├── data.html
├── training.html
├── inference.html
├── code-map.html
├── experiments.html
├── resources.html
├── styles.css
├── site.js
└── assets/
    ├── images/
    └── eval_images/
```

## 内容来源

教程基于 MiniMind-V 当前开源项目代码、README，以及 MiniMind-V / SigLIP2 / ALLaVA-4V 等公开资料整理。
