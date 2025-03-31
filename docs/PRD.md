# Interior Style Transfer POC using Flux on @fal

## 1. Executive Summary
This POC demonstrates an AI-driven interior design system that uses Flux via @fal to perform style transfer on input images of interiors. The goal is to enable three key transformations: style-only transfer, style+layout transfer, and furnishing of empty interiors—while preserving the room’s architectural structure. This is achieved via auto-segmentation and masking. A testable notebook will serve as the deliverable.

---

## 2. Problem Statement
Flux is capable of powerful generative style transformations, but results are often unrealistic when applied to real-world interiors due to structural inconsistencies (e.g., altered wall shapes, missing architectural context). A solution is needed that constrains Flux’s generation to within predefined structural masks while still enabling style-driven creativity.

---

## 3. Goals and Non-Goals

### Goals
- Develop a POC that:
  1. Applies a new style to an existing interior.
  2. Applies new style + layout (e.g., furniture repositioning).
  3. Furnishes an empty room in a target style.
- Preserves structural elements (walls, windows, floors) via masking.
- Demonstrates segmentation + generation pipeline.
- Uses Flux via the @fal API.
- Follows SSCS and TDD/BDD methodologies.

### Non-Goals
- This POC does not aim to deploy a production-ready service or provide a UI.
- Manual prompt engineering and tuning of styles are expected during development.

---

## 4. User Stories (BDD Style)

### User Persona
As a developer building interior design AI tools...

#### Story 1: Basic Style Transfer
**Given** an image of an interior  
**When** I apply a new style using Flux  
**Then** the style changes but the layout and structure remain the same

#### Story 2: Style + Layout Transfer
**Given** an image of an interior  
**When** I request a new style and layout  
**Then** the output reflects new furnishing positions while the structure stays intact

#### Story 3: Furnish Empty Room
**Given** an image of an empty room  
**When** I apply a style prompt  
**Then** the room is filled with styled interiors, without altering the architecture

#### Story 4: Prevent Structural Drift
**Given** any transformation  
**When** I use masking based on segmentation  
**Then** walls, windows, and architecture remain unchanged

---

## 5. Functional Requirements

| ID   | Requirement                                               | Priority | Test Case                             |
|------|-----------------------------------------------------------|----------|----------------------------------------|
| FR1  | Allow image upload or load from path                      | High     | Uploads test image successfully        |
| FR2  | Apply auto-segmentation to identify walls/floor/windows   | High     | Returns valid segmentation mask        |
| FR3  | Generate style-transferred image using Flux API           | High     | Image is returned from Flux endpoint   |
| FR4  | Mask structural elements to preserve architecture         | High     | Output image keeps original walls/shape|
| FR5  | Support prompts for style and layout change               | Medium   | Different prompts yield new layouts    |
| FR6  | Furnish empty room using prompt                           | Medium   | Empty room becomes styled interior     |
| FR7  | Display side-by-side input/output comparison              | Low      | Images displayed in output cells       |

---

## 6. Technical Requirements

### Stack
- **Language:** Python 3.11+
- **Environment:** Jupyter/Colab Notebook
- **APIs:** `@fal/ai`, `Flux`, `segment-anything`, `Grounding DINO` (optional)
- **Dependencies:** `torch`, `opencv`, `Pillow`, `matplotlib`, `fal`, `segment-anything`, `diffusers` (if needed)

### Structure


## 7. TDD Strategy

### Test Plan

| Test ID | Description                            | Inputs             | Expected Output                      |
|---------|----------------------------------------|--------------------|--------------------------------------|
| T1      | Test image upload                      | JPEG/PNG           | Valid image tensor                   |
| T2      | Test segmentation model loads correctly| SAM config         | Model object                         |
| T3      | Test mask generation                   | Image              | Binary mask of structure             |
| T4      | Test mask overlay works                | Image + mask       | Masked image preview                 |
| T5      | Test Flux API call                     | Image + prompt     | Styled output image                  |
| T6      | Test preservation of structure         | Input/output       | Visual similarity in structure       |
| T7      | Test empty room furnishing             | Empty room + prompt| Furnished output image               |

---

## 8. Acceptance Criteria
- At least one successful transformation for each of the 3 use cases.
- Structural masking is visibly preserved in outputs.
- Prompts generate visually distinct styles.
- Notebook is executable from top to bottom.
- API calls to @fal/Flux succeed and return outputs.

---

## 9. Risks and Mitigations

| Risk                                   | Mitigation                                          |
|----------------------------------------|-----------------------------------------------------|
| Flux introduces structural artifacts   | Use segmentation-based masking and prompt constraints |
| Segmentation fails on cluttered images| Use high-resolution input or Grounding DINO to assist |
| fal/Flux API is rate-limited           | Cache intermediate results, use offline Flux if needed |
| Output images appear low quality       | Adjust sampling steps, prompt tuning, or use mask-based control (ControlNet) |

---

## 10. Future Considerations
- Add UI for users to upload images and choose styles.
- Use fine-tuned models for specific styles (e.g., “Japandi”).
- Offer segmentation label selection for more control.
- Add diffusers-based local version of Flux if needed.
