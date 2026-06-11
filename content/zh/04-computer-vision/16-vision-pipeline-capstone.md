# 构建完整的视觉流水线 —— 毕业项目

> 生产级视觉系统是一条由数据契约串联起来的模型与规则之链。所需的组件在本阶段都已备齐；这个毕业项目要做的就是把它们端到端地连成一体。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 4 Lessons 01-15
**Time:** ~120 minutes

## 学习目标

- 设计一条生产级视觉流水线：检测目标、对其分类并输出结构化 JSON —— 同时处理好每一条失败路径
- 把检测器（Mask R-CNN 或 YOLO）、分类器（ConvNeXt-Tiny）和数据契约（Pydantic）接入同一个服务
- 对端到端流水线做基准测试，找出第一个瓶颈（通常是预处理，其次是检测器）
- 交付一个最小可用的 FastAPI 服务：接收图片上传、运行流水线、返回带分类结果的检测输出

## 问题背景

单个视觉模型有用；而视觉产品是它们的链条。零售货架巡检是检测器加商品分类器再加价格 OCR 流水线。自动驾驶是 2D 检测器加 3D 检测器加分割器加跟踪器加规划器。医疗预筛查是分割器加区域分类器再加面向临床医生的 UI。

把这些链条接起来，正是区分 ML 原型与产品的关键。模型之间的每一个接口都是新的 bug 滋生地。每一次坐标变换、每一次归一化、每一次掩码缩放，都是静默失败的候选点。流水线的强度取决于它最薄弱的接口。

这个毕业项目搭建的是最小可行流水线：检测 + 分类 + 结构化输出 + 服务层。Phase 4 中的其他内容都能插进这副骨架：把 Mask R-CNN 换成 YOLOv8、加一个 OCR 头、加一条分割分支、加一个跟踪器。架构是稳定的；组件是可插拔的。

## 核心概念

### 流水线

```mermaid
flowchart LR
    REQ["HTTP request<br/>+ image bytes"] --> LOAD["Decode<br/>+ preprocess"]
    LOAD --> DET["Detector<br/>(YOLO / Mask R-CNN)"]
    DET --> CROP["Crop + resize<br/>each detection"]
    CROP --> CLS["Classifier<br/>(ConvNeXt-Tiny)"]
    CLS --> AGG["Aggregate<br/>detections + classes"]
    AGG --> SCHEMA["Pydantic<br/>validation"]
    SCHEMA --> RESP["JSON response"]

    REQ -.->|error| RESP

    style DET fill:#fef3c7,stroke:#d97706
    style CLS fill:#dbeafe,stroke:#2563eb
    style SCHEMA fill:#dcfce7,stroke:#16a34a
```

一共七个阶段。两个模型阶段开销最大；其余五个阶段才是 bug 藏身之处。

### 用 Pydantic 定义数据契约

每个模型边界都变成一个带类型的对象。这能把静默失败变成显式报错。

```
Detection(
    box: tuple[float, float, float, float],   # (x1, y1, x2, y2), absolute pixels
    score: float,                              # [0, 1]
    class_id: int,                             # from detector's label map
    mask: Optional[list[list[int]]],           # RLE-encoded if present
)

PipelineResult(
    image_id: str,
    detections: list[Detection],
    classifications: list[Classification],
    inference_ms: float,
)
```

当检测器返回的框是 `(cx, cy, w, h)` 而不是 `(x1, y1, x2, y2)` 时，Pydantic 的校验会在边界处直接失败，你立刻就能发现问题，而不是去调试一个静默返回空区域的下游裁剪步骤。

### 延迟都花在哪里

几乎所有视觉流水线都满足三条规律：

1. **预处理往往是单项开销最大的环节。** 解码 JPEG、转换色彩空间、缩放尺寸 —— 这些都是 CPU 密集型操作，又最容易被忽略。
2. **检测器占据绝大部分 GPU 时间。** 70-90% 的 GPU 时间花在检测的前向计算上。
3. **后处理（NMS、RLE 编解码）在 GPU 上很便宜，在 CPU 上很昂贵。** 一定要在真实的目标硬件上做性能分析。

了解了时间分布，优化才能变成一份有优先级的清单。

### 失败模式

- **检测结果为空** —— 返回空列表，不要崩溃。记录日志。
- **框超出图像边界** —— 裁剪前把坐标钳制到图像尺寸以内。
- **裁剪区域过小** —— 对小于分类器最小输入尺寸的框跳过分类。
- **上传文件损坏** —— 返回带具体错误码的 400 响应，而不是 500。
- **模型加载失败** —— 在服务启动时就失败，而不是等到第一个请求才失败。

生产级流水线会逐一处理这些情况，而不是写一个掩盖失败的笼统 `try/except`。每一种失败都有命名的错误码和对应的响应。

### 批处理

生产服务要同时服务多个客户端。把多个请求的检测和分类合并成批，能成倍提升吞吐量。代价是：等待批次填满会增加额外延迟。典型配置：最多收集 20ms 的请求，合批处理，再分发响应。`torchserve` 和 `triton` 原生支持这种能力；负载可预测的小型服务则会自己实现一个微批处理器。

## 从零实现

### 第 1 步：数据契约

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple

class Detection(BaseModel):
    box: Tuple[float, float, float, float]
    score: float = Field(ge=0, le=1)
    class_id: int = Field(ge=0)
    mask_rle: Optional[str] = None


class Classification(BaseModel):
    detection_index: int
    class_id: int
    class_name: str
    score: float = Field(ge=0, le=1)


class PipelineResult(BaseModel):
    image_id: str
    detections: List[Detection]
    classifications: List[Classification]
    inference_ms: float
```

五秒钟写完的代码，能在任何正经的流水线上省下一小时的调试时间。

### 第 2 步：一个最小化的 Pipeline 类

```python
import time
import numpy as np
import torch
from PIL import Image

class VisionPipeline:
    def __init__(self, detector, classifier, class_names,
                 device="cpu", min_crop=32):
        self.detector = detector.to(device).eval()
        self.classifier = classifier.to(device).eval()
        self.class_names = class_names
        self.device = device
        self.min_crop = min_crop

    def preprocess(self, image):
        """
        image: PIL.Image or np.ndarray (H, W, 3) uint8
        returns: CHW float tensor on device
        """
        if isinstance(image, Image.Image):
            image = np.asarray(image.convert("RGB"))
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return tensor.to(self.device)

    @torch.no_grad()
    def detect(self, image_tensor):
        return self.detector([image_tensor])[0]

    @torch.no_grad()
    def classify(self, crops):
        if len(crops) == 0:
            return []
        batch = torch.stack(crops).to(self.device)
        logits = self.classifier(batch)
        probs = logits.softmax(-1)
        scores, cls = probs.max(-1)
        return list(zip(cls.tolist(), scores.tolist()))

    def run(self, image, image_id="anonymous"):
        t0 = time.perf_counter()
        tensor = self.preprocess(image)
        det = self.detect(tensor)

        crops = []
        detections = []
        valid_indices = []
        for i, (box, score, cls) in enumerate(zip(det["boxes"], det["scores"], det["labels"])):
            x1, y1, x2, y2 = [max(0, int(b)) for b in box.tolist()]
            x2 = min(x2, tensor.shape[-1])
            y2 = min(y2, tensor.shape[-2])
            detections.append(Detection(
                box=(x1, y1, x2, y2),
                score=float(score),
                class_id=int(cls),
            ))
            if (x2 - x1) < self.min_crop or (y2 - y1) < self.min_crop:
                continue
            crop = tensor[:, y1:y2, x1:x2]
            crop = torch.nn.functional.interpolate(
                crop.unsqueeze(0),
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            )[0]
            crops.append(crop)
            valid_indices.append(i)

        class_preds = self.classify(crops)

        classifications = []
        for valid_idx, (cls_id, cls_score) in zip(valid_indices, class_preds):
            classifications.append(Classification(
                detection_index=valid_idx,
                class_id=int(cls_id),
                class_name=self.class_names[cls_id],
                score=float(cls_score),
            ))

        return PipelineResult(
            image_id=image_id,
            detections=detections,
            classifications=classifications,
            inference_ms=(time.perf_counter() - t0) * 1000,
        )
```

每个接口都有类型。每条失败路径都有明确的处理决策。

### 第 3 步：接入检测器和分类器

```python
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models import convnext_tiny

# Use ImageNet-pretrained weights for a realistic pipeline without training
detector = maskrcnn_resnet50_fpn_v2(weights="DEFAULT")
classifier = convnext_tiny(weights="DEFAULT")
class_names = [f"imagenet_class_{i}" for i in range(1000)]

pipe = VisionPipeline(detector, classifier, class_names)

# Smoke test with a synthetic image
test_image = (np.random.rand(400, 600, 3) * 255).astype(np.uint8)
result = pipe.run(test_image, image_id="demo")
print(result.model_dump_json(indent=2)[:500])
```

### 第 4 步：FastAPI 服务

```python
from fastapi import FastAPI, UploadFile, HTTPException
from io import BytesIO

app = FastAPI()
pipe = None  # initialised on startup

@app.on_event("startup")
def load():
    global pipe
    detector = maskrcnn_resnet50_fpn_v2(weights="DEFAULT").eval()
    classifier = convnext_tiny(weights="DEFAULT").eval()
    pipe = VisionPipeline(detector, classifier, class_names=[f"c{i}" for i in range(1000)])

@app.post("/detect")
async def detect_endpoint(file: UploadFile):
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=400, detail="unsupported image type")
    data = await file.read()
    try:
        img = Image.open(BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="cannot decode image")
    result = pipe.run(img, image_id=file.filename or "upload")
    return result.model_dump()
```

用 `uvicorn main:app --host 0.0.0.0 --port 8000` 启动。用 `curl -F 'file=@dog.jpg' http://localhost:8000/detect` 测试。

### 第 5 步：对流水线做基准测试

```python
import time

def benchmark(pipe, num_runs=20, image_size=(400, 600)):
    img = (np.random.rand(*image_size, 3) * 255).astype(np.uint8)
    pipe.run(img)  # warm up

    stages = {"preprocess": [], "detect": [], "classify": [], "total": []}
    for _ in range(num_runs):
        t0 = time.perf_counter()
        tensor = pipe.preprocess(img)
        t1 = time.perf_counter()
        det = pipe.detect(tensor)
        t2 = time.perf_counter()
        crops = []
        for box in det["boxes"]:
            x1, y1, x2, y2 = [max(0, int(b)) for b in box.tolist()]
            x2 = min(x2, tensor.shape[-1])
            y2 = min(y2, tensor.shape[-2])
            if (x2 - x1) >= pipe.min_crop and (y2 - y1) >= pipe.min_crop:
                crop = tensor[:, y1:y2, x1:x2]
                crop = torch.nn.functional.interpolate(
                    crop.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
                )[0]
                crops.append(crop)
        pipe.classify(crops)
        t3 = time.perf_counter()
        stages["preprocess"].append((t1 - t0) * 1000)
        stages["detect"].append((t2 - t1) * 1000)
        stages["classify"].append((t3 - t2) * 1000)
        stages["total"].append((t3 - t0) * 1000)

    for stage, times in stages.items():
        times.sort()
        print(f"{stage:12s}  p50={times[len(times)//2]:7.1f} ms  p95={times[int(len(times)*0.95)]:7.1f} ms")
```

CPU 上的典型输出：preprocess 约 3 ms，detect 300-500 ms，classify 20-40 ms，total 350-550 ms。在 GPU 上，detect 降到 20-40 ms，预处理和分类的相对占比则开始变得更显著。

## 生产实践

生产环境的模板都会收敛到同一套结构，再加上：

- **模型版本管理** —— 始终在响应中记录模型名称和权重哈希。
- **按请求的追踪 ID（trace ID）** —— 为每个请求记录每个阶段的耗时，这样才能把慢响应与具体阶段关联起来。
- **降级路径** —— 如果分类器超时，返回不带分类结果的检测输出，而不是让整个请求失败。
- **安全过滤** —— NSFW / PII 过滤器在分类之后、响应离开服务之前运行。
- **批量端点** —— 提供一个接收图片 URL 列表的 `/detect_batch`，用于批量处理。

对于生产级服务部署，`torchserve`、`Triton Inference Server` 和 `BentoML` 开箱即用地提供批处理、版本管理、指标和健康检查。直接跑 `FastAPI` 对于原型和小规模产品来说完全够用。

## 交付产物

本课产出：

- `outputs/prompt-vision-service-shape-reviewer.md` —— 一个提示词，用于审查视觉服务代码中违反契约/响应结构的问题，并指出第一个会导致崩溃的 bug。
- `outputs/skill-pipeline-budget-planner.md` —— 一个技能：给定目标延迟和吞吐量，为流水线每个阶段分配时间预算，并标出哪个阶段会最先超出预算。

## 练习

1. **（简单）** 在任意开放数据集的 10 张图片上运行流水线。报告每个阶段的平均耗时，以及每张图片检测数量的分布。
2. **（中等）** 给 `Detection` 增加一个掩码输出字段，并用 RLE 编码。验证即使是包含 10 个目标的图片，JSON 也能保持在 1MB 以内。
3. **（困难）** 在分类器前面加一个微批处理器：最多收集 10 ms 的裁剪图，用一次 GPU 调用全部分类，再按请求返回结果。测量在每秒 5 个并发请求下的吞吐量提升以及新增的延迟。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------------|----------------------|
| 流水线（Pipeline） | "那个系统" | 由预处理、推理和后处理步骤组成的有序链条，每两个步骤之间都有带类型的接口 |
| 数据契约（Data contract） | "那个 schema" | Pydantic / dataclass 定义，每个阶段的输入和输出都必须符合；在边界处捕获集成 bug |
| 预处理（Preprocessing） | "模型之前的部分" | 解码、色彩转换、缩放、归一化；通常是最大的 CPU 时间消耗点 |
| 后处理（Postprocessing） | "模型之后的部分" | NMS、掩码缩放、阈值过滤、RLE 编码；GPU 上便宜，CPU 上昂贵 |
| 微批处理器（Microbatcher） | "先收集再前向" | 在固定时间窗内汇集多个请求，再执行一次合批前向计算的聚合器 |
| 追踪 ID（Trace ID） | "请求 id" | 在每个阶段都记录的按请求标识符，使慢请求可以被端到端追踪 |
| 失败码（Failure code） | "命名的错误" | 为每类失败定义具体错误码，而不是笼统的 500；让客户端的重试逻辑成为可能 |
| 健康检查（Health check） | "就绪探针" | 报告服务是否能正常应答的轻量端点；负载均衡器依赖它 |

## 延伸阅读

- [Full Stack Deep Learning — Deploying Models](https://fullstackdeeplearning.com/course/2022/lecture-5-deployment/) —— 生产环境 ML 部署的权威综述
- [BentoML docs](https://docs.bentoml.com) —— 提供批处理、版本管理和指标的服务部署框架
- [torchserve docs](https://pytorch.org/serve/) —— PyTorch 官方的服务部署库
- [NVIDIA Triton Inference Server](https://developer.nvidia.com/triton-inference-server) —— 支持批处理和多模型的高吞吐推理服务
