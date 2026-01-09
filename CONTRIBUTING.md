# Contributing to Xpectrass

Thank you for your interest in contributing to Xpectrass! We welcome contributions from the community.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue on GitHub with:
- A clear, descriptive title
- Steps to reproduce the behavior
- Expected behavior
- Actual behavior
- Your environment (OS, Python version, package versions)
- Code samples or test cases if applicable

### Suggesting Enhancements

Enhancement suggestions are welcome! Please create an issue with:
- A clear, descriptive title
- Detailed description of the proposed feature
- Use cases and examples
- Any relevant research papers or references

### Pull Requests

1. **Fork the repository** and create your branch from `main`
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Set up development environment**
   ```bash
   git clone https://github.com/kazilab/xpectrass.git
   cd xpectrass
   pip install -e ".[dev]"
   ```

3. **Make your changes**
   - Write clear, documented code
   - Follow PEP 8 style guidelines
   - Add docstrings to all functions/classes
   - Add type hints where appropriate

4. **Add tests**
   - Write tests for new functionality
   - Ensure all tests pass:
     ```bash
     pytest
     ```

5. **Update documentation**
   - Update relevant documentation files
   - Add examples if appropriate
   - Update CHANGELOG.md

6. **Format your code**
   ```bash
   black xpectrass/
   isort xpectrass/
   ```

7. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add: brief description of changes"
   ```

8. **Push to your fork and submit a pull request**
   ```bash
   git push origin feature/your-feature-name
   ```

## Development Guidelines

### Code Style

- Follow PEP 8
- Use Black for code formatting (line length: 88)
- Use isort for import sorting
- Write descriptive variable and function names
- Add docstrings in NumPy style

### Documentation

- Use NumPy-style docstrings
- Include type hints in function signatures
- Provide examples in docstrings when helpful
- Update user guide for significant features

### Testing

- Write unit tests for new features
- Aim for >80% code coverage
- Test edge cases and error conditions
- Use pytest fixtures for common setup

### Commit Messages

Use clear, descriptive commit messages:
- `Add: new feature description`
- `Fix: bug description`
- `Update: modification description`
- `Docs: documentation changes`
- `Test: test additions/modifications`

### Scientific Contributions

For new preprocessing methods or algorithms:
- Provide references to scientific papers
- Include validation against known datasets
- Document parameter recommendations
- Add examples to notebooks

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inspiring community for all. Please be respectful and constructive in your interactions.

### Our Standards

**Positive behaviors:**
- Being respectful of differing viewpoints
- Accepting constructive criticism gracefully
- Focusing on what is best for the community
- Showing empathy towards other community members

**Unacceptable behaviors:**
- Harassment or discriminatory language
- Trolling or insulting comments
- Publishing others' private information
- Other conduct inappropriate in a professional setting

## Questions?

Feel free to:
- Open an issue for questions
- Email us at xpectrass@kazilab.se
- Check our documentation at https://xpectrass.readthedocs.io/

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Xpectrass! 🔬📊
