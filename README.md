# CIFAR-10 Neural Network Comparison & AWS Inference Architecture

An end-to-end machine learning project comparing multiple neural network architectures for CIFAR-10 image classification and extending the highest-performing model into a containerized, cloud-based inference architecture on AWS.

The project progresses from model experimentation to API development, automated testing, containerization, cloud deployment, IAM-based security, and production monitoring.

## Results

The project evaluated multiple neural network architectures on the CIFAR-10 dataset, which contains 60,000 images across 10 classes.

| Model | Test Accuracy |
|---|---:|
| Basic MLP | 37.70% |
| ANN | 38.74% |
| DNN with Dropout | 34.54% |
| CNN | 68.14% |
| Fine-tuned MobileNetV2 | **87.63%** |

### Final MobileNetV2 Evaluation

Evaluated on the 10,000-image held-out test set:

- **Test Accuracy:** 87.63%
- **Weighted Precision:** 87.65%
- **Weighted Recall:** 87.63%
- **Weighted F1 Score:** 87.60%
- **Model Parameters:** ~2.42 million

## Architecture

The project separates model serving from the API layer:

**Client → FastAPI → Amazon SageMaker Serverless → MobileNetV2 → Prediction**

Supporting AWS infrastructure:

- **Amazon S3:** Private model artifact storage
- **Amazon SageMaker:** Serverless model inference
- **AWS IAM:** Scoped service permissions
- **Amazon CloudWatch:** Inference monitoring and observability
- **Amazon ECR:** Private FastAPI container registry
- **AWS CodeBuild:** Docker image build pipeline
- **GitHub Actions:** Automated API testing

The SageMaker Serverless endpoint was deployed, tested, benchmarked, and monitored during development, then removed after validation to avoid unnecessary ongoing cloud usage.
