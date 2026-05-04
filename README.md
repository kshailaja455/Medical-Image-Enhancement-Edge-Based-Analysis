# Medical Image Enhancement using Edge-Based Analysis

## 1. Overview
This project focuses on enhancing medical brain images such as CT scans and X-rays to improve visual interpretability. The system evaluates multiple image enhancement techniques and identifies the most effective method based on quantitative metrics.

## 2. Problem Statement
Medical images are typically grayscale, which makes it difficult to distinguish subtle differences in tissues. This can lead to:
1. Low contrast between structures   
2. Difficulty in identifying abnormalities  
3. Increased visual fatigue for medical professionals  

This project addresses the problem by applying and evaluating multiple enhancement techniques.

## 3. Dataset
1. Total Images: 2,734 medical brain scans  
2. Types: CT and X-ray images  
3. Source: Kaggle dataset  
4. Resolution: Standardized to 512 × 512  
5. Total Evaluations: 13,670 (5 methods per image) :contentReference[oaicite:3]{index=3}  

## 4. Methods Implemented
1. CLAHE (Contrast Limited Adaptive Histogram Equalization)  
2. Gamma Correction  
3. Heatmap (Jet Colormap)  
4. LUT Autumn Color Mapping  
5. Edge Enhancement using Canny Edge Detection  

## 5. Methodology
1. Image normalization and resizing  
2. Application of all enhancement techniques  
3. Evaluation using quantitative metrics  
4. Ranking based on composite score  
5. Identification of best method per image  

## 6. Evaluation Metrics
1. SSIM (Structural Similarity Index)  
2. PSNR (Peak Signal-to-Noise Ratio)  
3. Entropy (Information Content)  
4. Mean Pixel Difference  

## 7. Results
1. Edge Enhancement achieved highest performance  
2. Dominance: 92.54% of total images  
3. SSIM: ~0.96  
4. PSNR: ~31 dB  
5. Very low risk (0.88%) of misleading outputs :contentReference[oaicite:4]{index=4}  

## 8. Key Contributions
1. Developed a quantitative evaluation framework for medical image enhancement  
2. Implemented multiple enhancement techniques and compared their performance  
3. Designed a composite scoring system for objective ranking  
4. Identified Edge Enhancement as the most reliable method  

## 9. Authors
1. Ramesh Dadi  
2. Kanukuntla Shailaja  
3. Vallala Vyshnavi  

School of Computer Science and Artificial Intelligence  
SR University, Warangal  

# 10. Contributions

**Kanukuntla Shailaja:**

1. Image preprocessing (normalization, resizing)
2. Implementation of enhancement techniques (CLAHE, Gamma, Edge Enhancement)
3. Computation of evaluation metrics (SSIM, PSNR, Entropy, Mean Difference)

**Vallala Vyshnavi:**

1. Design of evaluation framework and workflow
2. Implementation of pseudo-coloring methods (Heatmap Jet, LUT Autumn)
3. Development of composite scoring mechanism

**Bhukya Madhu:**

1. Dataset collection and organization
2. Data formatting and pipeline setup
3. Management of input/output processing

**Nagarajuna Reddy Adapala:**

1. Statistical analysis and result validation
2. Comparative performance analysis across methods
3. Consistency evaluation of experimental outputs

**Vadalamani Veerabhadram:**

1. Visualization design (graphs, comparison outputs)
2. Documentation of results and report structuring
3. Preparation of figures and tables for presentation

**Ramesh Dadi (Supervisor):**

1. Project supervision and research guidance
2. Methodology validation and technical review
3. Final review of manuscript and research direction

## 11. Tech Stack
1. Python  
2. OpenCV  
3. NumPy  
4. Pandas  
5. Scikit-image  
6. Matplotlib  

## 12. How to Run
1. Install required libraries:
pip install numpy pandas opencv-python scikit-image matplotlib pillow
2. Run the script:
   python medical_framework.py --zip <dataset.zip>

## 13. Output Files Description

1. results.csv  
   Contains evaluation metrics for all images across all methods including SSIM, PSNR, Entropy, and Mean Difference.

2. ranked_results.csv  
   Contains results sorted by composite score for each image, showing method ranking.

3. best_method_per_image.csv  
   Shows the best-performing enhancement method for each image based on score.

4. summary_stats.csv  
   Provides average performance metrics for each method across the dataset.

5. potentially_misleading_cases.csv  
   Contains cases flagged as risky based on high entropy and low structural similarity.

6. Sample visualization outputs  
   Grid images showing comparison between original and enhanced outputs.  

## 14. Conclusion
This project demonstrates that edge-based enhancement is the most effective technique for preserving structural integrity while improving visual clarity in medical images.
