# Security Policy

## 🔒 Supported Versions

Currently supported versions with security updates:

| Version | Supported          |
|---------|---------------------|
| 1.x     | :white_check_mark:  |

## 🐛 Reporting a Vulnerability

**If you discover a security vulnerability, please DO NOT:**

- ❌ Open a public issue on GitHub
- ❌ Discuss it in public channels (Discord, Slack, etc.)
- ❌ Exploit the vulnerability

**Please DO:**

- ✅ Email us at furkankoykiran@gmail.com
- ✅ Include "VULNERABILITY" in the subject line
- ✅ Provide details about the vulnerability
- ✅ Include steps to reproduce (if applicable)

## 📧 What to Include

When reporting a vulnerability, please include:

1. **Description**: Clear description of the vulnerability
2. **Impact**: Potential impact if exploited
3. **Reproduction**: Steps to reproduce the issue
4. **Environment**: Version number, OS, Python version
5. **Proof of Concept** (if safe): Code or screenshot demonstrating the issue

## ⏱️ Response Timeline

- **Initial response**: Within 48 hours
- **Detailed response**: Within 7 days
- **Patch release**: As soon as feasible (typically within 14 days)
- **Security advisory**: Published with the patch

## 🔐 Security Best Practices

### For Users

1. **Keep dependencies updated**:
   ```bash
   pip install --upgrade -r requirements.txt
   ```

2. **Run in isolated environment**:
   - Use Docker containers
   - Use virtual environments
   - Don't run as root/admin

3. **Validate inputs**:
   - Check file types before upload
   - Limit file sizes
   - Sanitize user inputs

4. **Secure deployment**:
   - Use HTTPS in production
   - Add authentication
   - Implement rate limiting
   - Keep API keys private

### For Developers

1. **Input validation**:
   ```python
   # Validate file types
   ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
   if file.content_type not in ALLOWED_TYPES:
       raise HTTPException(400, "Invalid file type")
   ```

2. **File size limits**:
   ```python
   MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
   if file.size > MAX_FILE_SIZE:
       raise HTTPException(400, "File too large")
   ```

3. **Secure dependencies**:
   ```bash
   pip install safety
   safety check
   ```

4. **Code review**:
   - Review all code changes
   - Use static analysis tools
   - Run security scanners

## 🚨 Common Security Issues

### 1. Arbitrary File Upload

**Risk**: Users can upload malicious files

**Mitigation**:
```python
# Validate file type
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
def validate_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS
```

### 2. Model Inversion Attacks

**Risk**: Attackers can extract training data from the model

**Mitigation**:
- Don't expose raw model outputs in production
- Add noise to predictions
- Limit API access rate

### 3. DoS Attacks

**Risk**: Attackers can overwhelm the server with large files

**Mitigation**:
```python
# Implement rate limiting
from slowapi import Limiter

limiter = Limiter(key_func=get_remote_address)

@app.post("/detect")
@limiter.limit("10/minute")
async def detect(request: Request):
    ...
```

## 🔍 Security Audits

This project has NOT undergone a formal security audit. Use at your own risk.

For production deployments, consider:
- Professional security audit
- Penetration testing
- Regular dependency updates
- Monitoring and logging

## 📞 Contact

For security-related questions:
- Email: furkankoykiran@gmail.com
- GitHub Security: [Report vulnerability](https://github.com/furkankoykiran/ai-face-detector/security/advisories)

---

Thank you for helping keep AI Face Detector secure! 🔒
