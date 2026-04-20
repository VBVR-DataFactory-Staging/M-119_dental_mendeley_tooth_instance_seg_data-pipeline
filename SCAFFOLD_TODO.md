# M-119 scaffold TODO

Scaffolded from template: `M-038_isic2018_skin_lesion_seg_data-pipeline` (2026-04-20)

## Status
- [x] config.py updated (domain=dental_mendeley_tooth_instance_seg, s3_prefix=M-119_Dental-Mendeley/raw/, fps=3)
- [ ] core/download.py: update URL / Kaggle slug / HF repo_id
- [ ] src/download/downloader.py: adapt to dataset file layout
- [ ] src/pipeline/_phase2/*.py: adapt raw → frames logic (inherited from M-038_isic2018_skin_lesion_seg_data-pipeline, likely needs rework)
- [ ] examples/generate.py: verify end-to-end on 3 samples

## Task prompt
This panoramic dental X-ray (Mendeley). Segment each tooth as a distinct instance with unique color.

Fleet runs likely FAIL on first attempt for dataset parsing; iterate based on fleet logs at s3://vbvr-final-data/_logs/.
