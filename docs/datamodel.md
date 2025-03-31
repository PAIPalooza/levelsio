Here's a **detailed data model** for your **Interior Style Transfer POC** — designed for extensibility, testing, and SSCS-aligned modularity. Even though this is notebook-based, defining a solid **data model** helps maintain clarity, especially if you later convert it into a full API/backend service.

---

# 🧱 **Interior Style Transfer – Data Model**

---

## 📦 Core Entities

### 1. `InteriorImage`
Represents an input image of an interior space.

```python
@dataclass
class InteriorImage:
    id: str                      # Unique identifier
    file_path: str              # Local path or URL
    image_array: np.ndarray     # Raw image as NumPy array
    uploaded_at: datetime       # Timestamp
    source_type: str            # 'upload' | 'url' | 'test_sample'
```

---

### 2. `SegmentationMask`
Represents the mask used to preserve room construction (walls, ceiling, floor).

```python
@dataclass
class SegmentationMask:
    id: str
    source_image_id: str              # FK to InteriorImage
    mask_array: np.ndarray            # Binary mask
    segmentation_method: str          # e.g., 'SAM', 'GroundingDINO+SAM'
    classes_detected: List[str]       # Optional: ['wall', 'floor']
    created_at: datetime
```

---

### 3. `StylePrompt`
Encapsulates the user-provided or pre-set style description.

```python
@dataclass
class StylePrompt:
    id: str
    title: str                        # e.g., "Scandinavian Style"
    prompt_text: str                  # e.g., "a cozy Scandinavian style living room"
    allow_layout_change: bool         # True → allow layout to change
    applied_at: Optional[datetime]
```

---

### 4. `GenerationRequest`
Tracks each call made to Flux with all relevant data.

```python
@dataclass
class GenerationRequest:
    id: str
    input_image_id: str
    segmentation_mask_id: Optional[str]
    prompt_id: str
    model_used: str = "flux"
    fal_model_id: str = "fal-ai/flux"
    allow_layout_change: bool = False
    run_mode: str = "style_only"      # style_only | style_and_layout | furnish_empty
    created_at: datetime
    status: str = "pending"           # pending | complete | error
```

---

### 5. `GenerationResult`
Stores the results and metadata for each transformation.

```python
@dataclass
class GenerationResult:
    id: str
    generation_request_id: str
    output_image_array: np.ndarray
    output_path: Optional[str] = None
    similarity_score: Optional[float] = None  # (e.g., SSIM vs input)
    completed_at: datetime
```

---

## 📊 Optional: Metadata or Evaluation

### 6. `EvaluationResult`
For testing structural similarity or prompt effectiveness.

```python
@dataclass
class EvaluationResult:
    id: str
    input_image_id: str
    output_image_id: str
    ssim_score: float
    mse_score: float
    is_structure_preserved: bool
    human_rating: Optional[int]  # 1–5 scale for realism
    evaluated_at: datetime
```

---

## 📁 Suggested File Organization

```
project/
├── assets/
│   ├── test_images/
│   ├── outputs/
│   └── masks/
├── src/
│   ├── models.py          # Data models (dataclasses)
│   ├── segmentation.py    # SAM & masking functions
│   ├── generation.py      # fal/flux integration
│   ├── evaluation.py      # SSIM/MSE scoring
│   └── utils.py
├── notebooks/
│   └── interior-style-transfer.ipynb
├── tests/
│   └── test_pipeline.py
└── README.md
```

---

## 🔌 Usage Example in Notebook

```python
img = InteriorImage(id="img001", file_path="assets/test_images/room.jpg", ...)
mask = SegmentationMask(source_image_id=img.id, ...)
prompt = StylePrompt(title="Minimalist Loft", prompt_text="a clean modern minimalist loft")
req = GenerationRequest(input_image_id=img.id, prompt_id=prompt.id, ...)
```

---
