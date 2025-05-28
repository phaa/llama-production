# Image Processing API for Automated Utility Inspection

## Overview

This project is a scalable, GPU-accelerated image processing API built with **Python** and **FastAPI**, designed to automate utility inspection tasks using **vision-language models (LLaVA)**. The system transforms manual aerial image analysis into an AI-driven pipeline for faster, more consistent results.

---

## Technologies

- **FastAPI**: High-performance Python web framework for building asynchronous APIs.
- **Docker**: Containerized deployment with NVIDIA CUDA base images for GPU acceleration.
- **Hugging Face Transformers**: Integrated **LLaVA vision-to-text models** with 4-bit quantization via **BitsAndBytes** for optimized inference.
- **Google Cloud Vision OCR**: Optical Character Recognition service used to extract text from images.
- **PIL / OpenCV**: Image processing and manipulation.
- **Prometheus**: Real-time metrics collection.
- **Grafana**: Visualization of system metrics.
- **Uvicorn**: ASGI server for running FastAPI asynchronously with GPU support.

---

## Architecture

- **Microservice Design**: Asynchronous and scalable API endpoints.
- **GPU-Enabled Containers**: Deployed via Docker with NVIDIA support for accelerated model inference.
- **Orchestration**: Managed using **Docker Compose** for multi-container deployments.
- **Monitoring Stack**: **Prometheus** and **Grafana** for observability and performance tracking.

---

## Functionality

The API automates several aerial image annotation tasks:

### `/check-vegetation`
- Detects **vegetation and weed cover** in aerial images.
- Replaces manual identification to accelerate vegetation management.

### `/classify-switches`
- Identifies **pole-mounted disconnect switches**.
- Streamlines the detection of critical equipment on utility poles.

### `/process-transformer`
- Recognizes **pole-mounted transformers**.
- This endpoint performs a two-stage process:
  1. **Text Extraction**: Connects to the **Google Cloud Vision OCR** API to extract textual information from the transformer's nameplate.
  2. **Contextual Analysis**: Injects the extracted text into the **LLaVA model**, which generates structured **JSON** data based on the OCR output and visual context.
- Automates transformer identification and metadata extraction for infrastructure audits.

Each endpoint:
- Accepts uploaded images.
- Processes them using the appropriate model(s).
- Returns structured **JSON** outputs with detection and classification results.

---

## Problem Solving

The system addresses the challenge of **manual aerial image analysis** in utility inspections by introducing AI-driven automation for:

- **Vegetation detection**: Replacing labor-intensive weed and tree identification.
- **Equipment classification**: Automating the recognition of pole-mounted switches and transformers.
- **Text extraction and contextualization**: Using OCR combined with vision-language models to interpret and structure key information from equipment.

This solution improves:

- **Speed**: Real-time inference reduces field survey times.
- **Accuracy**: Consistent model-driven detection enhances reliability.
- **Scalability**: Asynchronous processing supports high-throughput operations.

---

## Results and Impact

- Achieved **real-time automated tagging** of utility imagery.
- Reduced **manual review time** and labor costs.
- Improved **consistency and accuracy** in infrastructure inspections.
- Enabled robust **system observability** with integrated monitoring using Prometheus and Grafana.

---

## Key Features

- Python-based **FastAPI** service for image processing.
- **GPU-accelerated** inference with **quantized LLaVA models**.
- Integration with **Google Cloud Vision OCR** for enhanced text extraction.
- **Dockerized microservice** architecture.
- Automated detection for **vegetation**, **overhead switches**, and **transformers**.
- Comprehensive **monitoring stack** for performance and reliability.

---

## Getting Started

1. Clone the repository.
2. Build the Docker images with GPU support.
3. Run the services using `docker-compose up`.
4. Access the API endpoints and monitoring dashboards.

---

## Monitoring

- Metrics exposed via **Prometheus** for automated scraping.
- Visualized in **Grafana** dashboards to ensure system reliability and performance.

---

## Keywords

**Python**, **FastAPI**, **Docker**, **LLM**, **Hugging Face**, **Google Cloud Vision**, **OCR**, **Computer Vision**, **Asynchronous Processing**, **Prometheus**, **GPU Acceleration**.

---

## Contributions

Feel free to fork the repository and submit pull requests. Suggestions and improvements are welcome.
