# Contributing to AI Face Detector

Thank you for your interest in contributing to AI Face Detector! This document provides guidelines and instructions for contributing.

## 🤝 How to Contribute

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When creating a bug report, include:

- **Clear title and description**: Describe the bug clearly
- **Steps to reproduce**: List the steps to reproduce the issue
- **Expected behavior**: What you expected to happen
- **Actual behavior**: What actually happened
- **Environment**: OS, Python version, dependencies
- **Screenshots/logs**: If applicable

### Suggesting Enhancements

Enhancement suggestions are welcome! Please:

- Use a clear and descriptive title
- Provide a detailed explanation of the enhancement
- Explain why this enhancement would be useful
- Include examples or mockups if applicable

### Pull Requests

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes using [conventional commits](https://www.conventionalcommits.org/)
   ```
   feat: add support for new image format
   fix: resolve memory leak in inference
   docs: update installation instructions
   ```
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### PR Guidelines

- **One feature per PR**: Keep changes focused
- **Follow code style**: Use the existing code style (Black formatting)
- **Add tests**: Include tests for new features
- **Update docs**: Update README and documentation
- **Small PRs**: Keep PRs small and focused for easier review

## 🛠️ Development Setup

### 1. Fork and Clone

```bash
git clone https://github.com/furkankoykiran/ai-face-detector.git
cd ai-face-detector
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Development Dependencies

```bash
pip install pytest pytest-asyncio httpx black ruff mypy
```

### 5. Run Tests

```bash
pytest tests/
```

### 6. Code Quality Checks

```bash
# Format code
black app/ training/

# Lint
ruff check app/ training/

# Type check
mypy app/
```

## 📁 Project Structure

```
AI-Face-Detector/
├── app/                    # FastAPI application
│   ├── __init__.py
│   ├── main.py            # FastAPI routes
│   ├── model_loader.py    # Model loading
│   ├── config.py          # Configuration
│   └── utils.py           # Utilities
├── training/              # Training scripts
│   ├── train_model.py     # Generic training
│   └── train_kaggle.py    # Kaggle-optimized training
├── static/                # Frontend files
│   └── index.html         # Web UI
├── docs/                  # Documentation
│   ├── API.md
│   └── TRAINING.md
└── tests/                 # Test files
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_api.py

# Run with coverage
pytest --cov=app tests/
```

### Writing Tests

- Use `pytest` for test framework
- Write descriptive test names
- Include edge cases
- Mock external dependencies

Example:
```python
def test_detect_endpoint_with_valid_image(client):
    response = client.post(
        "/detect",
        files={"file": ("test.jpg", open("test.jpg", "rb"), "image/jpeg")}
    )
    assert response.status_code == 200
    assert "result" in response.json()
```

## 📝 Documentation

### Updating README

When adding features:
1. Update the "Features" section
2. Add installation instructions if needed
3. Update usage examples
4. Add screenshots/GIFs for UI changes

### API Documentation

API docs are auto-generated from docstrings. Follow this format:

```python
def detect_face(file: UploadFile) -> JSONResponse:
    """
    Detect if a face image is real or AI-generated.

    Args:
        file: Uploaded image file (JPEG, PNG)

    Returns:
        JSON response with detection result and confidence

    Raises:
        400: Invalid file format
        500: Model not loaded
    """
```

## 🎨 Code Style

We follow these conventions:

- **Python**: PEP 8 with Black formatting
- **Imports**: Group imports (stdlib, third-party, local)
- **Docstrings**: Google style docstrings
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Line length**: Max 100 characters

## 🔐 Security

If you find a security vulnerability:

1. **Do NOT** open a public issue
2. Email us at furkankoykiran@gmail.com
3. Include details and reproduction steps
4. We'll respond within 48 hours

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Recognition

Contributors will be recognized in the CONTRIBUTORS.md file. Thank you for your contributions!

## 📧 Contact

- **Issues**: [GitHub Issues](https://github.com/furkankoykiran/ai-face-detector/issues)
- **Discussions**: [GitHub Discussions](https://github.com/furkankoykiran/ai-face-detector/discussions)
- **Email**: furkankoykiran@gmail.com

---

Happy contributing! 🚀
