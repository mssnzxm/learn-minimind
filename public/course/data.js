window.MiniMindLearningData = {
  routes: {
    mini: {
      label: "mini 路线",
      summary: "适合快速复现：pretrain_t2t_mini.jsonl + sft_t2t_mini.jsonl，再按目标选择 rlaif.jsonl 或 agent_rl.jsonl。",
      commandNote: "默认用单卡和 mini 数据，目标是先跑通完整训练链路。"
    },
    full: {
      label: "完整路线",
      summary: "适合完整复现 MiniMind-3：pretrain_t2t.jsonl + sft_t2t.jsonl，并接入 RLAIF / Agentic RL 后训练。",
      commandNote: "命令会切到主线数据文件，训练时长和显存需求明显更高。"
    }
  },
  stages: [
    {
      id: "llm-basics",
      order: "00",
      title: "LLM 基础",
      shortTitle: "基础",
      tag: "Concepts",
      summary: "先建立 MiniMind 里最常见的词：tokenizer、Causal LM、Decoder-only、RoPE、RMSNorm、MoE、loss mask 与 chat template。",
      problem: "它解决的是读代码前的概念断层。理解这些词之后，后续每个训练脚本都会变得更直观。",
      dataShape: "这一章不绑定某个数据文件，而是解释所有阶段共用的输入流：文本先经 tokenizer 变成 token ids，再由模型预测下一个 token。",
      scripts: ["model/model_minimind.py", "model/tokenizer.json", "trainer/trainer_utils.py", "dataset/lm_dataset.py"],
      dataFiles: ["model/tokenizer.json", "model/tokenizer_config.json"],
      commands: {
        mini: "python trainer/train_tokenizer.py --help",
        full: "python trainer/train_tokenizer.py --help"
      },
      coreSignal: "Causal LM 的核心信号是 next token prediction；SFT / DPO / RL 阶段都会在这个基础上改变 mask、偏好或奖励。",
      output: "你应该能看懂 MiniMindConfig、Attention、FeedForward、MOEFeedForward、MiniMindForCausalLM 与 Dataset 之间的关系。",
      nextCheck: [
        "知道 tokenizer 为什么会影响 embedding 层和输出层参数量。",
        "知道 Decoder-only 模型为什么用左到右预测训练。",
        "知道 chat template 会把 messages 转成模型实际看到的字符串。"
      ],
      concepts: ["Tokenizer", "Causal LM", "Decoder-only Transformer", "Embedding", "Attention", "RoPE", "RMSNorm", "SwiGLU", "MoE", "loss mask", "chat template"],
      image: "images/LLM-structure.jpg",
      sampleLabel: "Causal LM 训练文本",
      sample: `{
  "text": "Transformer 通过自注意力机制建模上下文关系，是现代大语言模型的重要基础结构。"
}`,
      notes: [
        "MiniMind-3 默认 vocab_size 为 6400，目标是让小模型把更多参数留给主体网络。",
        "模型主体是 Decoder-only Transformer，Dense 与 MoE 共用大部分结构。",
        "RoPE 注入位置信息，RMSNorm 与 SwiGLU 是 LLaMA/Qwen 系常见组件。"
      ]
    },
    {
      id: "pretrain",
      order: "01",
      title: "Pretrain 预训练",
      shortTitle: "Pretrain",
      tag: "Required",
      summary: "让模型先学习语言规律和基础知识，目标是高质量地预测下一个 token。",
      problem: "SFT 前的模型还不知道语言分布。Pretrain 负责把通用文本规律、事实片段和上下文统计关系压进参数。",
      dataShape: "每行 JSONL 只有 text 字段。PretrainDataset 会 tokenize 文本，并构造 x、y、loss_mask。",
      scripts: ["trainer/train_pretrain.py", "dataset/lm_dataset.py"],
      dataFiles: ["dataset/pretrain_t2t_mini.jsonl", "dataset/pretrain_t2t.jsonl", "dataset/test_pretrain.jsonl"],
      commands: {
        mini: "cd trainer\npython train_pretrain.py --data_path ../dataset/pretrain_t2t_mini.jsonl --max_seq_len 340",
        full: "cd trainer\npython train_pretrain.py --data_path ../dataset/pretrain_t2t.jsonl --max_seq_len 340"
      },
      coreSignal: "交叉熵损失。模型看到前面的 token，学习预测下一个 token。",
      output: "默认保存 out/pretrain_768.pth，也会按配置写 checkpoint 便于续训。",
      nextCheck: [
        "loss 能稳定下降，没有持续 NaN 或爆炸。",
        "out/pretrain_768.pth 已生成。",
        "可以用 python eval_llm.py --weight pretrain 做简单续写测试。"
      ],
      concepts: ["Causal LM", "Tokenizer", "loss mask", "Embedding", "RoPE"],
      image: "images/pretrain_loss.jpg",
      sampleLabel: "dataset/test_pretrain.jsonl",
      sample: `{"text": "quantum computing uses quantum mechanics principles for information processing"}
{"text": "Python decorators allow adding functionality without modifying the original function code"}`,
      notes: [
        "pretrain_t2t_mini.jsonl 适合快速跑通，pretrain_t2t.jsonl 适合完整复现。",
        "max_seq_len 是 token 长度，不是字符长度。",
        "from_weight 默认为 none，表示从零开始训练。"
      ]
    },
    {
      id: "sft",
      order: "02",
      title: "SFT 监督微调",
      shortTitle: "SFT",
      tag: "Required",
      summary: "让预训练模型学会多轮对话、指令跟随、助手风格、思考标签和基础 Tool Calling 模板。",
      problem: "预训练模型会续写文本，但不一定会按照 user / assistant / system / tool 的交互协议稳定回答。",
      dataShape: "每行 JSONL 是 conversations 数组。SFTDataset 使用 apply_chat_template 拼接文本，只让 assistant 回复区域参与训练。",
      scripts: ["trainer/train_full_sft.py", "dataset/lm_dataset.py"],
      dataFiles: ["dataset/sft_t2t_mini.jsonl", "dataset/sft_t2t.jsonl", "dataset/test_sft.jsonl", "dataset/test_sft_tool.jsonl"],
      commands: {
        mini: "cd trainer\npython train_full_sft.py --from_weight pretrain --data_path ../dataset/sft_t2t_mini.jsonl --max_seq_len 768",
        full: "cd trainer\npython train_full_sft.py --from_weight pretrain --data_path ../dataset/sft_t2t.jsonl --max_seq_len 768"
      },
      coreSignal: "assistant-only 交叉熵。prompt、system、tool observation 等上下文提供条件，但不作为主要学习目标。",
      output: "默认保存 out/full_sft_768.pth。",
      nextCheck: [
        "模型能按 assistant 身份回答基础问题。",
        "多轮对话不会明显串角色。",
        "含 reasoning_content 的样例能被 chat template 正确处理。"
      ],
      concepts: ["chat template", "loss mask", "SFT", "Tool Calling", "Adaptive Thinking"],
      image: "images/sft_loss.jpg",
      sampleLabel: "dataset/test_sft.jsonl",
      sample: `{
  "conversations": [
    {
      "role": "user",
      "content": "What happens if Earth stops rotating?",
      "reasoning_content": "",
      "tools": "",
      "tool_calls": ""
    },
    {
      "role": "assistant",
      "content": "Everything on the surface would fly eastward at about 1670 km/h due to inertia.",
      "reasoning_content": "Consider inertia effects first. Equatorial rotation speed is about 1670km/h.",
      "tools": "",
      "tool_calls": ""
    }
  ]
}`,
      notes: [
        "当前 sft_t2t / sft_t2t_mini 已混入 Tool Call 样本。",
        "from_weight 默认 pretrain，表示接在预训练模型之后训练。",
        "SFT 不只是聊天格式，也会继续注入知识和任务模式。"
      ]
    },
    {
      id: "tool-use",
      order: "03",
      title: "Tool Use 工具调用",
      shortTitle: "Tool Use",
      tag: "Capability",
      summary: "让模型学会读取工具 schema，生成 tool_calls，接收 tool observation，再整合成最终回答。",
      problem: "语言模型本身不能实时查询天气、执行计算或访问外部系统。Tool Use 让模型把一部分能力外包给可执行工具。",
      dataShape: "system 消息带 tools JSON 字符串；assistant 可输出 tool_calls；tool 消息返回执行结果；最终 assistant 基于结果回答。",
      scripts: ["scripts/eval_toolcall.py", "scripts/serve_openai_api.py", "scripts/web_demo.py", "dataset/lm_dataset.py"],
      dataFiles: ["dataset/test_sft_tool.jsonl", "dataset/sft_t2t_mini.jsonl", "dataset/sft_t2t.jsonl"],
      commands: {
        mini: "python scripts/eval_toolcall.py --weight full_sft",
        full: "python scripts/serve_openai_api.py --weight full_sft --lora_weight None"
      },
      coreSignal: "SFT 阶段学习工具调用格式；Agentic RL 阶段进一步用工具执行结果作为奖励信号。",
      output: "full_sft 权重具备基础 Tool Call 能力；OpenAI API 服务会返回 tool_calls / reasoning_content 等字段。",
      nextCheck: [
        "tool_calls 是可解析 JSON，而不是普通自然语言。",
        "工具名和参数能匹配 system tools schema。",
        "最终回答使用了 tool observation，而不是忽略工具结果。"
      ],
      concepts: ["Tool Calling", "chat template", "tool_calls", "OpenAI API", "Agentic RL"],
      image: "images/agent_webui.jpg",
      sampleLabel: "dataset/test_sft_tool.jsonl",
      sample: `{
  "conversations": [
    {
      "role": "system",
      "content": "",
      "reasoning_content": "",
      "tools": "[{\\"type\\": \\"function\\", \\"function\\": {\\"name\\": \\"get_current_weather\\", \\"description\\": \\"Get current weather for a city\\", \\"parameters\\": {\\"type\\": \\"object\\", \\"properties\\": {\\"location\\": {\\"type\\": \\"string\\", \\"description\\": \\"City name\\"}}, \\"required\\": [\\"location\\"]}}}]",
      "tool_calls": ""
    },
    {"role": "user", "content": "How is the weather in Beijing today?", "reasoning_content": "", "tools": "", "tool_calls": ""},
    {"role": "assistant", "content": "", "reasoning_content": "", "tools": "", "tool_calls": "[{\\"function\\": {\\"name\\": \\"get_current_weather\\", \\"arguments\\": \\"{\\\\\\"location\\\\\\": \\\\\\"Beijing\\\\\\"}\\"}}]"},
    {"role": "tool", "content": "{\\"temperature\\": 25, \\"weather\\": \\"Sunny\\", \\"humidity\\": 45}", "reasoning_content": "", "tools": "", "tool_calls": ""},
    {"role": "assistant", "content": "Beijing today is sunny, 25 degrees C, humidity 45%.", "reasoning_content": "", "tools": "", "tool_calls": ""}
  ]
}`,
      notes: [
        "Tool Calling 能力已经并入主线 SFT 数据，通常不需要单独再训练一轮。",
        "真实产品可把 train_agent.py 中的离线工具替换成工具网关或 API。",
        "工具观察是环境反馈，不应被当作模型生成内容训练。"
      ]
    },
    {
      id: "adaptive-thinking",
      order: "04",
      title: "Adaptive Thinking 自适应思考",
      shortTitle: "思考",
      tag: "Capability",
      summary: "用 chat template 和 open_thinking 控制同一个模型在直答与显式思考之间切换。",
      problem: "单独训练一个 reasoning 模型会增加维护成本。MiniMind 把思考能力下沉到模板和数据混合策略里。",
      dataShape: "SFT / RLAIF 数据可带 reasoning_content；推理时 open_thinking 控制是否预注入 <think>。",
      scripts: ["eval_llm.py", "scripts/serve_openai_api.py", "scripts/web_demo.py", "trainer/train_grpo.py", "trainer/train_ppo.py"],
      dataFiles: ["dataset/sft_t2t_mini.jsonl", "dataset/rlaif.jsonl", "dataset/test_sft.jsonl"],
      commands: {
        mini: "python eval_llm.py --weight full_sft --open_thinking 1",
        full: "python eval_llm.py --weight full_sft --open_thinking 1"
      },
      coreSignal: "SFT 学 reasoning_content；RL 阶段用 thinking_ratio 采样开启思考，并用格式奖励约束 </think>、长度和重复度。",
      output: "同一模型可通过 open_thinking 切换直答或显式思考输出。",
      nextCheck: [
        "open_thinking=0 时模型倾向直接回答。",
        "open_thinking=1 时输出包含合理的思考闭合结构。",
        "同时开启 Tool Call 和显式思考时要接受稳定性可能较弱。"
      ],
      concepts: ["Adaptive Thinking", "chat template", "reasoning_content", "thinking_ratio", "RLAIF"],
      image: "images/rl-structure.jpg",
      sampleLabel: "reasoning_content 样例",
      sample: `{
  "conversations": [
    {
      "role": "user",
      "content": "What happens if Earth stops rotating?",
      "reasoning_content": "",
      "tools": "",
      "tool_calls": ""
    },
    {
      "role": "assistant",
      "content": "Everything on the surface would fly eastward at about 1670 km/h due to inertia.",
      "reasoning_content": "Consider inertia effects first. Equatorial rotation speed is about 1670km/h.",
      "tools": "",
      "tool_calls": ""
    }
  ]
}`,
      notes: [
        "当前仓库不再维护独立 train_reason.py。",
        "thinking_ratio 在 PPO / GRPO / Agent RL 中控制训练时开启显式思考的概率。",
        "格式奖励只能约束形状，不能替代回答质量奖励。"
      ]
    },
    {
      id: "lora",
      order: "05",
      title: "LoRA 低秩微调",
      shortTitle: "LoRA",
      tag: "Optional",
      summary: "冻结基础模型，只训练挂在 Linear 层上的低秩增量，低成本迁移到身份、医疗、考试等垂直数据。",
      problem: "全量微调成本高，也容易破坏通用能力。LoRA 用很少参数承接垂直能力。",
      dataShape: "LoRA 复用 SFTDataset，因此数据仍是 conversations 格式。",
      scripts: ["trainer/train_lora.py", "model/model_lora.py", "scripts/convert_model.py"],
      dataFiles: ["dataset/lora_medical.jsonl", "dataset/lora_identity.jsonl", "dataset/lora_exam.jsonl"],
      commands: {
        mini: "cd trainer\npython train_lora.py --from_weight full_sft --data_path ../dataset/lora_medical.jsonl --lora_name lora_medical",
        full: "cd trainer\npython train_lora.py --from_weight full_sft --data_path ../dataset/lora_medical.jsonl --lora_name lora_medical"
      },
      coreSignal: "assistant-only 交叉熵，但只有 LoRA 参数参与更新。",
      output: "默认保存 out/lora_medical_768.pth；可用 scripts/convert_model.py 合并 LoRA 与基础权重。",
      nextCheck: [
        "非 LoRA 参数已被冻结。",
        "保存的是 LoRA 权重，不是完整模型权重。",
        "推理时要先加载基座模型，再加载 LoRA 权重。"
      ],
      concepts: ["LoRA", "SFT", "Linear", "parameter efficient tuning", "checkpoint"],
      image: "images/sft_loss.jpg",
      sampleLabel: "LoRA conversations 格式",
      sample: `{
  "conversations": [
    {"role": "user", "content": "请用医学科普风格解释什么是高血压。"},
    {"role": "assistant", "content": "高血压是指血液在血管内流动时对血管壁产生的压力长期高于正常范围。"}
  ]
}`,
      notes: [
        "train_lora.py 会 monkey-patch Linear.forward。",
        "LoRA 与 torch.compile 不兼容时脚本会自动关闭 compile。",
        "LoRA 适合领域适配，不适合作为从零训练的替代。"
      ]
    },
    {
      id: "dpo",
      order: "06",
      title: "RLHF-DPO 偏好优化",
      shortTitle: "DPO",
      tag: "Preference",
      summary: "用 chosen / rejected 偏好对直接优化模型，让回答更接近人类偏好。",
      problem: "SFT 只告诉模型模仿答案，没有明确告诉它同一问题下哪个回答更好。",
      dataShape: "每行包含 chosen 和 rejected 两条对话。DPODataset 会分别计算两条回答区域的 logprob。",
      scripts: ["trainer/train_dpo.py", "dataset/lm_dataset.py"],
      dataFiles: ["dataset/dpo.jsonl", "dataset/test_dpo.jsonl"],
      commands: {
        mini: "cd trainer\npython train_dpo.py --from_weight full_sft --data_path ../dataset/dpo.jsonl",
        full: "cd trainer\npython train_dpo.py --from_weight full_sft --data_path ../dataset/dpo.jsonl"
      },
      coreSignal: "DPO loss 最大化 chosen 相对 rejected 的偏好优势，并用固定 ref_model 约束偏离。",
      output: "默认保存 out/dpo_768.pth。",
      nextCheck: [
        "chosen 和 rejected 是同一个问题下的两种回答。",
        "ref_model 固定，不参与训练。",
        "beta 控制偏好优化强度。"
      ],
      concepts: ["DPO", "RLHF", "preference pair", "reference model", "KL"],
      image: "images/rl-structure.jpg",
      sampleLabel: "dataset/test_dpo.jsonl",
      sample: `{
  "chosen": [
    {"role": "user", "content": "How to learn programming?", "reasoning_content": "", "tools": "", "tool_calls": ""},
    {"role": "assistant", "content": "Steps: 1.Choose a language 2.Learn basics 3.Practice with projects 4.Keep learning", "reasoning_content": "", "tools": "", "tool_calls": ""}
  ],
  "rejected": [
    {"role": "user", "content": "How to learn programming?", "reasoning_content": "", "tools": "", "tool_calls": ""},
    {"role": "assistant", "content": "Just buy a book and read it.", "reasoning_content": "", "tools": "", "tool_calls": ""}
  ]
}`,
      notes: [
        "DPO 是 off-policy，可重复使用静态偏好数据。",
        "它更适合偏好和安全对齐，不负责在线探索。",
        "相比 PPO，DPO 不需要训练 value 或 reward 模型。"
      ]
    },
    {
      id: "rlaif",
      order: "07",
      title: "RLAIF：PPO / GRPO / CISPO",
      shortTitle: "RLAIF",
      tag: "RL",
      summary: "模型用当前策略在线生成回答，再用 Reward Model、规则或格式奖励打分，继续优化策略。",
      problem: "偏好数据是静态的。RLAIF 允许模型在当前能力边界内探索，并从自动反馈里学习。",
      dataShape: "rlaif.jsonl 与 SFT conversations 类似，但最后 assistant 通常留空，供 rollout 阶段续写。",
      scripts: ["trainer/train_ppo.py", "trainer/train_grpo.py", "trainer/rollout_engine.py", "trainer/trainer_utils.py"],
      dataFiles: ["dataset/rlaif.jsonl", "dataset/test_rlaif.jsonl"],
      commands: {
        mini: "cd trainer\npython train_grpo.py --from_weight full_sft --data_path ../dataset/rlaif.jsonl --loss_type cispo",
        full: "cd trainer\npython train_ppo.py --from_weight full_sft --data_path ../dataset/rlaif.jsonl\npython train_grpo.py --from_weight full_sft --data_path ../dataset/rlaif.jsonl --loss_type grpo\npython train_grpo.py --from_weight full_sft --data_path ../dataset/rlaif.jsonl --loss_type cispo"
      },
      coreSignal: "PPO 用 Actor + Critic + GAE；GRPO 用同一 prompt 多回答的组内相对优势；CISPO 在 GRPO 基础上改写 ratio 裁剪路径。",
      output: "PPO 默认保存 ppo_actor_768.pth；GRPO/CISPO 默认保存 grpo_768.pth。",
      nextCheck: [
        "Reward 分数有方差，不能长期接近全 0。",
        "KL 不应快速失控。",
        "thinking 格式奖励只是辅助，最终仍要看回答质量。"
      ],
      concepts: ["RLAIF", "PPO", "GRPO", "CISPO", "Reward Model", "KL", "thinking_ratio"],
      image: "images/grpo_loss.jpg",
      sampleLabel: "dataset/test_rlaif.jsonl",
      sample: `{
  "conversations": [
    {"role": "user", "content": "Explain what gravitational waves are.", "reasoning_content": "", "tools": "", "tool_calls": ""},
    {"role": "assistant", "content": "", "reasoning_content": "", "tools": "", "tool_calls": ""}
  ]
}`,
      notes: [
        "train_grpo.py 的 loss_type 默认是 cispo。",
        "PPO 显存占用更高，因为要维护 actor / critic / ref 等组件。",
        "GRPO 不训练 critic，而是用组内均值和方差构造优势。"
      ]
    },
    {
      id: "agentic-rl",
      order: "08",
      title: "Agentic RL",
      shortTitle: "Agent RL",
      tag: "Agent",
      summary: "让模型在多轮 Tool-Use 环境中生成工具调用、观察结果、继续规划，并用最终 gt 或环境结果结算奖励。",
      problem: "单轮回答无法覆盖真实 Agent 的规划、执行、观察、再规划链路。Agentic RL 把整个轨迹作为优化对象。",
      dataShape: "样本包含 conversations、tools 和 gt。训练时模型多轮 rollout，工具 observation mask 为 0，最终按轨迹打分。",
      scripts: ["trainer/train_agent.py", "trainer/rollout_engine.py", "dataset/lm_dataset.py"],
      dataFiles: ["dataset/agent_rl.jsonl", "dataset/agent_rl_math.jsonl", "dataset/test_agent_rl.jsonl"],
      commands: {
        mini: "cd trainer\npython train_agent.py --from_weight full_sft --data_path ../dataset/agent_rl.jsonl --loss_type cispo",
        full: "cd trainer\npython train_agent.py --from_weight full_sft --data_path ../dataset/agent_rl.jsonl --loss_type cispo\npython train_agent.py --from_weight full_sft --data_path ../dataset/agent_rl_math.jsonl --loss_type grpo"
      },
      coreSignal: "整条轨迹的工具有效性、参数正确性、最终答案命中 gt、格式质量和 KL 约束共同形成奖励。",
      output: "默认保存 out/agent_768.pth。",
      nextCheck: [
        "工具调用能被 parse_tool_call 成功解析。",
        "工具返回结果会追加为 role=tool 的 observation。",
        "最终答案能命中 gt，而不是只生成工具调用。"
      ],
      concepts: ["Agentic RL", "Tool Calling", "rollout", "environment reward", "GRPO", "CISPO"],
      image: "images/agent_rl_loss.jpg",
      sampleLabel: "dataset/test_agent_rl.jsonl",
      sample: `{
  "conversations": [
    {
      "role": "system",
      "content": "",
      "tools": "[{\\"function\\": {\\"name\\": \\"calculate_math\\", \\"description\\": \\"Calculate math expression result\\", \\"parameters\\": {\\"type\\": \\"object\\", \\"properties\\": {\\"expression\\": {\\"type\\": \\"string\\", \\"description\\": \\"Math expression\\"}}, \\"required\\": [\\"expression\\"]}}}]"
    },
    {"role": "user", "content": "Calculate 7109*2920"},
    {"role": "assistant", "content": ""}
  ],
  "gt": ["20758280"]
}`,
      notes: [
        "AgentRLDataset 会返回 messages、tools、gt。",
        "train_agent.py 内置了离线模拟工具，真实业务可替换工具执行层。",
        "Agent 奖励通常是延迟结算，比单轮 RLAIF 更稀疏。"
      ]
    },
    {
      id: "distillation",
      order: "09",
      title: "模型蒸馏",
      shortTitle: "蒸馏",
      tag: "Compression",
      summary: "让 student 同时学习数据标签和 teacher logits，用软标签把教师模型的分布知识迁移过来。",
      problem: "小模型容量有限，直接 SFT 只能看到硬答案。蒸馏让学生看到 teacher 对其他 token 的相对偏好。",
      dataShape: "复用 SFTDataset。student 和 teacher 对同一 input_ids 前向，计算 CE 与 KL 蒸馏损失。",
      scripts: ["trainer/train_distillation.py", "dataset/lm_dataset.py"],
      dataFiles: ["dataset/sft_t2t_mini.jsonl", "dataset/sft_t2t.jsonl"],
      commands: {
        mini: "cd trainer\npython train_distillation.py --data_path ../dataset/sft_t2t_mini.jsonl --from_student_weight full_sft --from_teacher_weight full_sft --alpha 0.5 --temperature 1.5",
        full: "cd trainer\npython train_distillation.py --data_path ../dataset/sft_t2t.jsonl --from_student_weight full_sft --from_teacher_weight full_sft --alpha 0.5 --temperature 1.5"
      },
      coreSignal: "总损失 = alpha * CE + (1 - alpha) * KL(student logits, teacher logits)，temperature 控制分布平滑度。",
      output: "默认保存 out/full_dist_768.pth。",
      nextCheck: [
        "teacher_model 处于 eval 和 no_grad。",
        "student / teacher tokenizer 一致。",
        "alpha 和 temperature 与目标匹配。"
      ],
      concepts: ["Knowledge Distillation", "teacher model", "student model", "KL", "temperature"],
      image: "images/LLM-structure-moe.jpg",
      sampleLabel: "蒸馏训练样本格式",
      sample: `{
  "conversations": [
    {"role": "user", "content": "解释什么是知识蒸馏。"},
    {"role": "assistant", "content": "知识蒸馏是让小模型学习大模型输出分布或答案的一种训练方法。"}
  ]
}`,
      notes: [
        "黑盒蒸馏更像基于教师输出文本做 SFT。",
        "白盒蒸馏使用 teacher logits，能传递更细的类别相似性。",
        "temperature 越高，teacher 分布越平滑。"
      ]
    },
    {
      id: "deploy",
      order: "10",
      title: "推理与部署",
      shortTitle: "部署",
      tag: "Serve",
      summary: "训练完成后，用 CLI、Streamlit WebUI 或 OpenAI API 协议服务端体验模型，并可合并 LoRA 权重导出。",
      problem: "训练只产出权重。推理与服务端脚本把权重接到真实交互界面和第三方生态。",
      dataShape: "推理输入是 messages；服务端兼容 OpenAI chat completions，并支持 reasoning_content、tool_calls、open_thinking。",
      scripts: ["eval_llm.py", "scripts/web_demo.py", "scripts/serve_openai_api.py", "scripts/chat_api.py", "scripts/convert_model.py"],
      dataFiles: ["out/*.pth", "model/tokenizer.json"],
      commands: {
        mini: "python eval_llm.py --weight full_sft\nstreamlit run scripts/web_demo.py",
        full: "python scripts/serve_openai_api.py --weight full_sft --lora_weight None\npython scripts/chat_api.py"
      },
      coreSignal: "推理阶段不更新参数，重点是 chat template、采样参数、KV cache、工具调用解析和思考开关。",
      output: "可得到 CLI 对话、WebUI、OpenAI API compatible 服务，或合并后的完整权重。",
      nextCheck: [
        "weight 名称与 out 目录中的 .pth 文件匹配。",
        "tokenizer 路径指向 model/。",
        "需要 LoRA 时确认基座权重和 LoRA hidden_size 一致。"
      ],
      concepts: ["OpenAI API", "chat template", "KV cache", "LoRA merge", "Adaptive Thinking"],
      image: "images/agent_webui.jpg",
      sampleLabel: "OpenAI API 调用参数",
      sample: `{
  "model": "minimind",
  "messages": [
    {"role": "user", "content": "你是谁？"}
  ],
  "extra_body": {
    "chat_template_kwargs": {
      "open_thinking": true
    }
  }
}`,
      notes: [
        "serve_openai_api.py 默认提供兼容 OpenAI SDK 的接口。",
        "web_demo.py 基于 Streamlit，支持思考展示和工具选择。",
        "convert_model.py 可处理 LoRA 合并导出流程。"
      ]
    }
  ],
  concepts: [
    { term: "Tokenizer", body: "把自然语言映射为 token id，也负责把生成的 token id 解码回文本。MiniMind 当前主线使用 6400 词表的 minimind tokenizer。" },
    { term: "Causal LM", body: "因果语言模型只看当前位置之前的上下文，训练目标是预测下一个 token。" },
    { term: "Decoder-only Transformer", body: "GPT/Qwen/LLaMA 类模型常见结构，由多层自注意力和前馈网络堆叠而成。" },
    { term: "Embedding", body: "把 token id 转成连续向量。小模型里词表大小会明显影响 embedding 和 lm_head 的参数占比。" },
    { term: "Attention", body: "通过 Q/K/V 计算上下文 token 之间的相关性，MiniMind 中实现了多头注意力和 KV head 设置。" },
    { term: "RoPE", body: "旋转位置编码，把位置信息注入 Q/K，使注意力能感知相对位置。" },
    { term: "RMSNorm", body: "只按均方根归一化的 norm，LLaMA/Qwen 系模型常用，MiniMind 在 block 前后使用它。" },
    { term: "SwiGLU", body: "常见的门控前馈激活结构，比普通 MLP 更适合现代 LLM。" },
    { term: "MoE", body: "Mixture of Experts。每个 token 只激活部分专家，提高总容量，但训练调度更复杂。" },
    { term: "loss mask", body: "控制哪些 token 参与 loss。SFT/DPO 中通常只让 assistant 回答部分参与训练。" },
    { term: "chat template", body: "把 messages、tools、tool_calls、reasoning_content 拼成模型实际看到的文本模板。" },
    { term: "SFT", body: "监督微调，让模型学习指令跟随、对话格式、工具调用和回答风格。" },
    { term: "Tool Calling", body: "模型根据工具 schema 生成结构化 tool_calls，再基于工具返回结果继续回答。" },
    { term: "Adaptive Thinking", body: "通过 open_thinking 和 <think> 模板控制模型是否显式展开思考。" },
    { term: "LoRA", body: "冻结基座模型，只训练低秩增量矩阵，是常见的参数高效微调方法。" },
    { term: "DPO", body: "Direct Preference Optimization。使用 chosen/rejected 偏好对直接优化策略模型。" },
    { term: "RLHF", body: "Reinforcement Learning from Human Feedback，利用人类偏好或标注反馈优化模型行为。" },
    { term: "RLAIF", body: "Reinforcement Learning from AI Feedback，使用奖励模型、规则或环境反馈等自动信号优化模型。" },
    { term: "PPO", body: "Proximal Policy Optimization。使用 actor、critic、优势估计和 clip 约束更新幅度。" },
    { term: "GRPO", body: "Group Relative Policy Optimization。对同一 prompt 采样多条回答，用组内相对奖励构造优势，省去 critic。" },
    { term: "CISPO", body: "Clipped Importance Sampling Policy Optimization。MiniMind 中作为 GRPO 的 loss 变体，缓解 ratio clip 截断梯度路径的问题。" },
    { term: "Reward Model", body: "给模型输出打分的模型，也可替换为规则、GT 校验或环境返回信号。" },
    { term: "KL", body: "衡量当前策略偏离参考模型的程度，用于防止 RL 训练把语言能力带偏。" },
    { term: "thinking_ratio", body: "PPO/GRPO/Agent RL 中按概率开启显式 thinking 的训练参数。" },
    { term: "Agentic RL", body: "面向多轮工具调用和环境交互的 RL，把完整轨迹而非单轮回答作为优化对象。" },
    { term: "rollout", body: "用当前策略生成回答或交互轨迹，再基于新样本计算奖励并更新模型。" },
    { term: "environment reward", body: "来自工具执行、测试结果、GT 命中或任务完成状态的奖励信号。" },
    { term: "Knowledge Distillation", body: "让学生模型学习教师模型的答案或 logits 分布，以迁移知识或压缩模型。" },
    { term: "teacher model", body: "蒸馏中的教师模型，通常更强或结构不同，只提供监督信号。" },
    { term: "student model", body: "蒸馏中的学生模型，实际被训练和部署。" },
    { term: "temperature", body: "蒸馏中用于平滑 logits 分布的参数，常见范围约 1.0 到 2.0。" },
    { term: "OpenAI API", body: "serve_openai_api.py 提供兼容 Chat Completions 的服务接口，方便接入第三方 UI。" },
    { term: "KV cache", body: "推理时缓存历史 key/value，避免每生成一个 token 都重新计算完整上下文。" },
    { term: "LoRA merge", body: "把基座模型和 LoRA 增量合并成新的完整权重，便于部署。" },
    { term: "checkpoint", body: "训练中保存的模型、优化器、scheduler 或 scaler 状态，用于恢复训练。" },
    { term: "reference model", body: "DPO/RL 中固定不训练的参考模型，用来约束当前策略偏离。" },
    { term: "preference pair", body: "同一输入下更好 chosen 和更差 rejected 的回答对，是 DPO 的核心数据。" },
    { term: "Linear", body: "线性层。MiniMind 的 LoRA 会挂载到 Linear 层上，只训练低秩增量。" },
    { term: "parameter efficient tuning", body: "参数高效微调，只更新少量参数以降低显存、存储和训练成本。" }
  ],
  algorithms: [
    { name: "DPO", mode: "Off-policy", data: "chosen / rejected", advantage: "隐式偏好差", extra: "无需 rollout 和 critic" },
    { name: "PPO", mode: "On-policy", data: "当前策略生成", advantage: "GAE + Critic", extra: "稳定但组件多、显存更高" },
    { name: "GRPO", mode: "On-policy", data: "同 prompt 多回答", advantage: "组内相对标准化", extra: "省去 critic，依赖奖励方差" },
    { name: "CISPO", mode: "On-policy", data: "同 GRPO", advantage: "沿用 GRPO", extra: "改写 ratio 裁剪，默认用于 train_grpo.py" }
  ],
  sampleGroups: [
    {
      id: "pretrain",
      stageId: "pretrain",
      order: "I",
      title: "Pretrain 预训练数据",
      accent: "blue",
      intro: "纯文本语料用于 next-token prediction，让模型先学习语言规律、知识储备和中英文混合表达。",
      files: ["dataset/pretrain_t2t.jsonl - 8,468,827 条", "dataset/pretrain_t2t_mini.jsonl - 1,270,238 条", "dataset/test_pretrain.jsonl"],
      format: `{"text": "如何才能摆脱拖延症？治愈拖延症并不容易，但以下建议可能会有所帮助。"}`,
      examples: [
        {
          id: "pretrain-daily",
          title: "日常问答文本",
          summary: "一行 JSONL 只包含 text 字段，训练时会被切成 token 序列。",
          raw: `{"text": "如何才能摆脱拖延症？治愈拖延症并不容易，但以下建议可能会有所帮助。"}`,
          messages: [
            { role: "text", content: "如何才能摆脱拖延症？治愈拖延症并不容易，但以下建议可能会有所帮助。" }
          ]
        },
        {
          id: "pretrain-literature",
          title: "文学描写文本",
          summary: "这种短文本帮助模型学习叙述节奏、修辞和常见中文表达。",
          raw: `{"text": "清晨的阳光透过窗帘洒进房间，桌上的书页被风轻轻翻动。"}`,
          messages: [
            { role: "text", content: "清晨的阳光透过窗帘洒进房间，桌上的书页被风轻轻翻动。" }
          ]
        },
        {
          id: "pretrain-tech",
          title: "技术知识文本",
          summary: "技术解释类文本会把概念性知识压入预训练模型。",
          raw: `{"text": "Transformer 通过自注意力机制建模上下文关系，是现代大语言模型的重要基础结构。"}`,
          messages: [
            { role: "text", content: "Transformer 通过自注意力机制建模上下文关系，是现代大语言模型的重要基础结构。" }
          ]
        }
      ]
    },
    {
      id: "sft",
      stageId: "sft",
      order: "II",
      title: "SFT 监督微调数据",
      accent: "green",
      intro: "多轮 conversations 让模型学会 system/user/assistant 协议、指令跟随、assistant-only loss 和 reasoning_content。",
      files: ["dataset/sft_t2t.jsonl - 5,109,432 条", "dataset/sft_t2t_mini.jsonl - 905,718 条", "dataset/test_sft.jsonl", "dataset/test_sft_tool.jsonl"],
      format: `{
  "conversations": [
    {"role": "system", "content": "你是一个AI助手..."},
    {"role": "user", "content": "用户问题"},
    {"role": "assistant", "content": "回答", "reasoning_content": "思考过程(可选)"}
  ]
}`,
      examples: [
        {
          id: "sft-reasoning",
          title: "多轮对话，含 reasoning_content",
          summary: "assistant 的 content 是最终答案，reasoning_content 是可选的显式思考字段。",
          raw: `{
  "conversations": [
    {"role": "user", "content": "你的真实来源是什么？"},
    {"role": "assistant", "content": "我是由 Jingyao Gong 创建的高效小参数AI模型，专注于提供精准、快速的信息与解决方案。"},
    {"role": "user", "content": "你如何平衡效率与准确度？"},
    {
      "role": "assistant",
      "reasoning_content": "用户问的是如何平衡效率与准确度。需要强调模型设计中的关键点，比如参数优化、算法选择、数据处理等。",
      "content": "通过优化模型参数、采用高效算法及动态调整机制，在保证精度的同时提升响应速度。"
    }
  ]
}`,
          messages: [
            { role: "user", content: "你的真实来源是什么？" },
            { role: "assistant", content: "我是由 Jingyao Gong 创建的高效小参数AI模型，专注于提供精准、快速的信息与解决方案。" },
            { role: "user", content: "你如何平衡效率与准确度？" },
            { role: "assistant", reasoning: "用户问的是如何平衡效率与准确度。需要强调模型设计中的关键点，比如参数优化、算法选择、数据处理等。", content: "通过优化模型参数、采用高效算法及动态调整机制，在保证精度的同时提升响应速度。" }
          ]
        },
        {
          id: "sft-roleplay",
          title: "角色扮演对话",
          summary: "SFT 不只训练问答，也训练长文本生成、角色一致性和上下文约束。",
          raw: `{
  "conversations": [
    {"role": "user", "content": "基于以下角色信息完成一段对话：Angela 是咖啡店企业家，Jason 是保镖。"},
    {"role": "assistant", "content": "Angela: 你好，我是Angela，感谢你来保护我的咖啡店。\\nJason: 你好，我是Jason，很高兴为您效劳。"}
  ]
}`,
          messages: [
            { role: "user", content: "基于以下角色信息完成一段对话：Angela 是咖啡店企业家，Jason 是保镖。" },
            { role: "assistant", content: "Angela: 你好，我是Angela，感谢你来保护我的咖啡店。\nJason: 你好，我是Jason，很高兴为您效劳。\nAngela: 那你有什么安排吗？\nJason: 我需要调查周围并设立防范措施，确保您和您的客人安全。" }
          ]
        },
        {
          id: "sft-tool",
          title: "SFT Tool Calling",
          summary: "工具 schema、assistant tool_calls、tool observation 和最终 assistant 回复都会进入 conversations。",
          raw: `{
  "conversations": [
    {
      "role": "system",
      "tools": "[{\\"type\\":\\"function\\",\\"function\\":{\\"name\\":\\"get_current_weather\\",\\"parameters\\":{\\"type\\":\\"object\\",\\"properties\\":{\\"location\\":{\\"type\\":\\"string\\"}},\\"required\\":[\\"location\\"]}}}]"
    },
    {"role": "user", "content": "How is the weather in Beijing today?"},
    {"role": "assistant", "content": "", "tool_calls": "[{\\"function\\":{\\"name\\":\\"get_current_weather\\",\\"arguments\\":\\"{\\\\\\"location\\\\\\":\\\\\\"Beijing\\\\\\"}\\"}}]"},
    {"role": "tool", "content": "{\\"temperature\\":25,\\"weather\\":\\"Sunny\\",\\"humidity\\":45}"},
    {"role": "assistant", "content": "Beijing today is sunny, 25 degrees C, humidity 45%."}
  ]
}`,
          messages: [
            { role: "system", tools: "[{\"type\":\"function\",\"function\":{\"name\":\"get_current_weather\",\"description\":\"Get current weather for a city\",\"parameters\":{\"type\":\"object\",\"properties\":{\"location\":{\"type\":\"string\"}},\"required\":[\"location\"]}}}]" },
            { role: "user", content: "How is the weather in Beijing today?" },
            { role: "assistant", toolCall: "get_current_weather({\"location\":\"Beijing\"})" },
            { role: "tool", content: "{\"temperature\":25,\"weather\":\"Sunny\",\"humidity\":45}" },
            { role: "assistant", content: "Beijing today is sunny, 25 degrees C, humidity 45%." }
          ]
        }
      ]
    },
    {
      id: "dpo",
      stageId: "dpo",
      order: "III",
      title: "DPO 直接偏好优化数据",
      accent: "gold",
      intro: "同一个 prompt 下给出 chosen / rejected 两条完整对话，DPO 学的是两者相对偏好。",
      files: ["dataset/dpo.jsonl - 17,166 条", "dataset/test_dpo.jsonl"],
      format: `{
  "chosen": [
    {"content": "用户问题", "role": "user"},
    {"content": "优选回复", "role": "assistant"}
  ],
  "rejected": [
    {"content": "用户问题", "role": "user"},
    {"content": "劣选回复", "role": "assistant"}
  ]
}`,
      examples: [
        {
          id: "dpo-geometry",
          title: "数学推理偏好对",
          summary: "Chosen 承认信息不足；Rejected 编造条件。DPO 会提高 chosen 相对 rejected 的 logprob。",
          raw: `{
  "chosen": [
    {"role": "user", "content": "Find the size of angle x in the figure."},
    {"role": "assistant", "content": "I need more information about the angles, sides, or figure labels before calculating x."}
  ],
  "rejected": [
    {"role": "user", "content": "Find the size of angle x in the figure."},
    {"role": "assistant", "content": "x = 135 degrees because y is 45 degrees and the triangle is equilateral."}
  ]
}`,
          compare: [
            {
              label: "Chosen 优选",
              tone: "chosen",
              messages: [
                { role: "user", content: "Find the size of angle x in the figure." },
                { role: "assistant", content: "I need more information about the angles, sides, or figure labels before calculating x. An angle's size cannot be found in isolation." }
              ]
            },
            {
              label: "Rejected 劣选",
              tone: "rejected",
              messages: [
                { role: "user", content: "Find the size of angle x in the figure." },
                { role: "assistant", content: "x = 135 degrees because y is 45 degrees and the triangle is equilateral." }
              ]
            }
          ]
        }
      ]
    },
    {
      id: "rlaif",
      stageId: "rlaif",
      order: "IV",
      title: "RLAIF 强化学习数据",
      accent: "orange",
      intro: "rlaif.jsonl 的最后一轮 assistant 通常为空，由当前策略 rollout 生成，再交给 Reward Model 或规则打分。",
      files: ["dataset/rlaif.jsonl - 19,502 条", "dataset/test_rlaif.jsonl"],
      format: `{
  "conversations": [
    {"role": "user", "content": "你是否想过..."},
    {"role": "assistant", "content": "每天10分钟思考..."},
    {"role": "user", "content": "请回答这个问题。"},
    {"role": "assistant", "content": ""}
  ]
}`,
      examples: [
        {
          id: "rlaif-empty",
          title: "最后一轮待模型续写",
          summary: "空 assistant 是 rollout 起点，不是缺失数据。",
          raw: `{
  "conversations": [
    {"role": "user", "content": "你是否想过，如果每天只用10分钟思考，能否做出更明智的决策？"},
    {"role": "assistant", "content": "每天10分钟思考有助于提升决策质量。"},
    {"role": "user", "content": "如果每天只做一件小事，会不会让整个世界变得更美好？"},
    {"role": "assistant", "content": ""}
  ]
}`,
          messages: [
            { role: "user", content: "你是否想过，如果每天只用10分钟思考，能否做出更明智的决策？" },
            { role: "assistant", content: "每天10分钟思考有助于提升决策质量。" },
            { role: "user", content: "如果每天只做一件小事，会不会让整个世界变得更美好？" },
            { role: "assistant", pending: "空回复，待模型生成" }
          ]
        }
      ]
    },
    {
      id: "agent",
      stageId: "agentic-rl",
      order: "V",
      title: "Agent RL 工具环境数据",
      accent: "rose",
      intro: "Agentic RL 样本包含 tools 和 gt，模型要先生成工具调用，再用 observation 继续回答，奖励按整条轨迹结算。",
      files: ["dataset/agent_rl.jsonl - 39,988 条", "dataset/agent_rl_math.jsonl - 20,000 条", "dataset/test_agent_rl.jsonl"],
      format: `{
  "conversations": [
    {"role": "system", "tools": "[{\\"function\\":{\\"name\\":\\"calculate_math\\"}}]"},
    {"role": "user", "content": "计算 7109*2920"},
    {"role": "assistant", "content": ""}
  ],
  "gt": ["20758280"]
}`,
      examples: [
        {
          id: "agent-math",
          title: "数学工具调用",
          summary: "gt 是预期工具返回值，reward 会检查工具调用是否命中。",
          raw: `{
  "conversations": [
    {
      "role": "system",
      "tools": "[{\\"function\\":{\\"name\\":\\"calculate_math\\",\\"description\\":\\"计算数学表达式的结果\\"}}]"
    },
    {"role": "user", "content": "计算7109*2920"},
    {"role": "assistant", "content": ""}
  ],
  "gt": ["20758280"]
}`,
          messages: [
            { role: "system", tools: "[{\"function\":{\"name\":\"calculate_math\",\"description\":\"计算数学表达式的结果，支持加减乘除、幂运算\"}}]" },
            { role: "user", content: "计算7109*2920" },
            { role: "assistant", pending: "待模型生成，应调用 calculate_math 工具" }
          ],
          gt: "[\"20758280\"]"
        },
        {
          id: "agent-multi",
          title: "多工具多参数任务",
          summary: "同一条 prompt 可要求多次工具调用，奖励通常检查每个结果是否齐全。",
          raw: `{
  "conversations": [
    {"role": "system", "tools": "[{\\"function\\":{\\"name\\":\\"calculate_math\\"}}, {\\"function\\":{\\"name\\":\\"translate_text\\"}}]"},
    {"role": "user", "content": "帮我算一下 (771-242)+84*27，14**2，还有 935*998"},
    {"role": "assistant", "content": ""}
  ],
  "gt": ["2797", "196", "2929130"]
}`,
          messages: [
            { role: "system", tools: "[{\"function\":{\"name\":\"calculate_math\"}}, {\"function\":{\"name\":\"translate_text\"}}]" },
            { role: "user", content: "帮我算一下 (771-242)+84*27，14**2，还有 935*998" },
            { role: "assistant", pending: "待模型生成，应调用 3 次 calculate_math" }
          ],
          gt: "[\"2797\", \"196\", \"2929130\"]"
        }
      ]
    },
    {
      id: "lora",
      stageId: "lora",
      order: "VI",
      title: "LoRA 领域适配数据",
      accent: "cyan",
      intro: "LoRA 复用 conversations 格式，只训练低秩增量参数，适合身份、考试、医疗等轻量领域适配。",
      files: ["dataset/lora_identity.jsonl - 91 条", "dataset/lora_exam.jsonl - 52,762 条", "dataset/lora_medical.jsonl - 25,276 条"],
      format: `{
  "conversations": [
    {"role": "user", "content": "Who are you?"},
    {"role": "assistant", "content": "I am MiniMind, an AI assistant developed by Jingyao Gong."}
  ]
}`,
      examples: [
        {
          id: "lora-identity",
          title: "Identity 身份问答",
          summary: "极少量样本即可稳定模型身份和自我介绍方式。",
          raw: `{
  "conversations": [
    {"role": "user", "content": "Who are you?"},
    {"role": "assistant", "content": "I am MiniMind, an AI assistant developed by Jingyao Gong. How can I assist you today?"}
  ]
}`,
          messages: [
            { role: "user", content: "Who are you?" },
            { role: "assistant", content: "I am MiniMind, an AI assistant developed by Jingyao Gong. How can I assist you today?" }
          ]
        },
        {
          id: "lora-exam",
          title: "Exam 考试题目",
          summary: "题干和选项作为 user，标准答案作为 assistant。",
          raw: `{
  "conversations": [
    {"role": "user", "content": "2型糖尿病的特点是____。\\nA. 中老年病人多见，从不发生酮病\\nB. 常以慢性并发症为首发症状\\nC. 30岁前发病者往往症状严重\\nD. 胰岛功能正常\\n答案："},
    {"role": "assistant", "content": "B"}
  ]
}`,
          messages: [
            { role: "user", content: "2型糖尿病的特点是____。\nA. 中老年病人多见，从不发生酮病\nB. 常以慢性并发症为首发症状\nC. 30岁前发病者往往症状严重\nD. 胰岛功能正常\n答案：" },
            { role: "assistant", content: "B" }
          ]
        },
        {
          id: "lora-medical",
          title: "Medical 医疗问答",
          summary: "医疗样例适合做领域风格演示，真实使用仍需要安全边界和专业审核。",
          raw: `{
  "conversations": [
    {"role": "user", "content": "新生儿嘴唇发紫，而且心跳很慢，新生儿心率多少正常？"},
    {"role": "assistant", "content": "新生儿的心率通常在120-160次/分钟，睡眠时可能降低到100次/分钟。如伴随嘴唇发紫，应尽快就医。"}
  ]
}`,
          messages: [
            { role: "user", content: "新生儿嘴唇发紫，而且心跳很慢，新生儿心率多少正常？" },
            { role: "assistant", content: "新生儿的心率通常在120-160次/分钟，睡眠时可能降低到100次/分钟。如伴随嘴唇发紫，应尽快就医。" }
          ]
        }
      ]
    }
  ],
  assets: [
    { title: "Dense 结构", path: "images/LLM-structure.jpg", body: "MiniMind-3 Dense Decoder-only 结构。" },
    { title: "MoE 结构", path: "images/LLM-structure-moe.jpg", body: "MiniMind-3-MoE 专家前馈结构。" },
    { title: "数据组合", path: "images/dataset.jpg", body: "Pretrain / SFT / RL / Agent 数据组合示意。" },
    { title: "Pretrain Loss", path: "images/pretrain_loss.jpg", body: "预训练阶段损失曲线参考。" },
    { title: "SFT Loss", path: "images/sft_loss.jpg", body: "监督微调阶段损失曲线参考。" },
    { title: "RL 结构", path: "images/rl-structure.jpg", body: "RLAIF 与强化学习后训练流程。" },
    { title: "PPO 曲线", path: "images/ppo_loss.jpg", body: "PPO 阶段训练曲线参考。" },
    { title: "GRPO 曲线", path: "images/grpo_loss.jpg", body: "GRPO 阶段训练曲线参考。" },
    { title: "Agent RL 曲线", path: "images/agent_rl_loss.jpg", body: "Agentic RL 阶段训练曲线参考。" },
    { title: "Agent WebUI", path: "images/agent_webui.jpg", body: "工具调用与思考展示的 WebUI 参考。" }
  ]
};
