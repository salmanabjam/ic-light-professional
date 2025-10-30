"""
AI Dev Collective Agent Roles Implementation
Each agent has specific expertise and responsibilities
"""

import os
import ast
import re
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from .base import Agent, AgentReport


class AstroLeadDeveloper(Agent):
    """
    Lead Developer - Astro
    Responsibilities:
    - Leads software development, coding, architecture design
    - Implements core features, ensures scalability and maintainability
    - Ensures code meets best practices and is optimized for performance
    """
    
    def __init__(self):
        super().__init__(
            name="Astro",
            role="Lead Developer",
            responsibilities=[
                "Leads software development, including coding, architecture design, and integration.",
                "Implements all core features and ensures the software is scalable and maintainable.",
                "Ensures code meets best practices and is optimized for performance."
            ]
        )
    
    def analyze(self, project_path: str) -> AgentReport:
        """Analyze code architecture and development practices"""
        report = AgentReport(
            agent_name=self.name,
            role=self.role,
            timestamp=datetime.now().isoformat()
        )
        
        # Analyze Python files
        py_files = list(Path(project_path).rglob("*.py"))
        
        total_lines = 0
        total_functions = 0
        total_classes = 0
        files_analyzed = 0
        
        for py_file in py_files:
            if '.git' in str(py_file) or '__pycache__' in str(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    total_lines += len(lines)
                    
                    # Parse AST
                    tree = ast.parse(content)
                    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
                    classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
                    
                    total_functions += len(functions)
                    total_classes += len(classes)
                    files_analyzed += 1
            except:
                continue
        
        # Findings
        report.findings.append(f"Analyzed {files_analyzed} Python files")
        report.findings.append(f"Total lines of code: {total_lines}")
        report.findings.append(f"Total functions: {total_functions}")
        report.findings.append(f"Total classes: {total_classes}")
        
        # Check for key architecture files
        has_init = (Path(project_path) / "__init__.py").exists()
        has_requirements = (Path(project_path) / "requirements.txt").exists()
        has_readme = (Path(project_path) / "README.md").exists()
        
        if has_requirements:
            report.findings.append("✅ Requirements file found")
        else:
            report.findings.append("⚠️ No requirements.txt file found")
            
        if has_readme:
            report.findings.append("✅ README documentation found")
        else:
            report.findings.append("⚠️ No README.md file found")
        
        # Recommendations
        report.recommendations.append("Consider modular architecture with clear separation of concerns")
        report.recommendations.append("Implement comprehensive error handling across all modules")
        report.recommendations.append("Add type hints to improve code maintainability")
        
        if total_functions > 0:
            avg_functions_per_file = total_functions / files_analyzed
            if avg_functions_per_file > 10:
                report.recommendations.append(f"Consider splitting files with many functions (avg: {avg_functions_per_file:.1f} per file)")
        
        # Metrics
        report.metrics = {
            "files_analyzed": files_analyzed,
            "total_lines": total_lines,
            "total_functions": total_functions,
            "total_classes": total_classes,
            "avg_lines_per_file": total_lines / files_analyzed if files_analyzed > 0 else 0,
            "has_requirements": has_requirements,
            "has_readme": has_readme
        }
        
        report.priority = "high"
        return report


class LyraResearchAssistant(Agent):
    """
    Research Assistant - Lyra
    Responsibilities:
    - Investigates new technologies, frameworks, and tools
    - Provides recommendations on emerging trends
    - Evaluates libraries and dependencies
    """
    
    def __init__(self):
        super().__init__(
            name="Lyra",
            role="Research Assistant",
            responsibilities=[
                "Investigates new technologies, frameworks, and tools that could enhance the software.",
                "Provides recommendations on how to incorporate emerging trends and technologies.",
                "Assists in evaluating libraries and third-party dependencies for improvements."
            ]
        )
    
    def analyze(self, project_path: str) -> AgentReport:
        """Analyze dependencies and suggest improvements"""
        report = AgentReport(
            agent_name=self.name,
            role=self.role,
            timestamp=datetime.now().isoformat()
        )
        
        # Check requirements file
        req_file = Path(project_path) / "requirements.txt"
        dependencies = []
        
        if req_file.exists():
            with open(req_file, 'r') as f:
                dependencies = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
            report.findings.append(f"Found {len(dependencies)} dependencies in requirements.txt")
            
            # Check for key ML/AI libraries
            key_libs = {
                'torch': 'PyTorch for deep learning',
                'transformers': 'Hugging Face Transformers',
                'diffusers': 'Diffusers for image generation',
                'gradio': 'Gradio for UI',
                'opencv': 'OpenCV for image processing',
                'pillow': 'PIL for image handling',
                'numpy': 'NumPy for numerical computing'
            }
            
            found_libs = []
            for dep in dependencies:
                dep_lower = dep.lower()
                for lib, desc in key_libs.items():
                    if lib in dep_lower:
                        found_libs.append(f"{lib}: {desc}")
            
            if found_libs:
                report.findings.append("Key libraries detected:")
                report.findings.extend([f"  - {lib}" for lib in found_libs])
        else:
            report.findings.append("⚠️ No requirements.txt file found")
        
        # Recommendations
        report.recommendations.append("Evaluate latest versions of dependencies for security updates")
        report.recommendations.append("Consider adding pytest for automated testing")
        report.recommendations.append("Explore MLflow for experiment tracking")
        report.recommendations.append("Consider using pre-commit hooks for code quality")
        report.recommendations.append("Investigate wandb for advanced ML monitoring")
        
        report.metrics = {
            "total_dependencies": len(dependencies),
            "has_requirements_file": req_file.exists()
        }
        
        report.priority = "medium"
        return report


class NexusCodeQualityAssistant(Agent):
    """
    Code Quality Assistant - Nexus
    Responsibilities:
    - Reviews codebase for quality and readability
    - Suggests improvements for maintainability and performance
    - Ensures code follows industry standards
    """
    
    def __init__(self):
        super().__init__(
            name="Nexus",
            role="Code Quality Assistant",
            responsibilities=[
                "Reviews the existing codebase for quality and readability.",
                "Suggests improvements to improve maintainability and performance.",
                "Ensures code is optimized, follows industry standards, and is free of bugs."
            ]
        )
    
    def analyze(self, project_path: str) -> AgentReport:
        """Analyze code quality metrics"""
        report = AgentReport(
            agent_name=self.name,
            role=self.role,
            timestamp=datetime.now().isoformat()
        )
        
        py_files = list(Path(project_path).rglob("*.py"))
        
        code_issues = []
        long_functions = []
        missing_docstrings = []
        
        for py_file in py_files:
            if '.git' in str(py_file) or '__pycache__' in str(py_file):
                continue
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    
                    # Check for very long lines
                    long_lines = [i+1 for i, line in enumerate(lines) if len(line) > 120]
                    if long_lines:
                        code_issues.append(f"{py_file.name}: {len(long_lines)} lines exceed 120 chars")
                    
                    # Parse for docstrings
                    try:
                        tree = ast.parse(content)
                        for node in ast.walk(tree):
                            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                                if not ast.get_docstring(node):
                                    missing_docstrings.append(f"{py_file.name}:{node.lineno} - {node.name}")
                            
                            # Check function length
                            if isinstance(node, ast.FunctionDef):
                                func_length = node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 0
                                if func_length > 50:
                                    long_functions.append(f"{py_file.name}:{node.name} ({func_length} lines)")
                    except:
                        pass
            except:
                continue
        
        # Findings
        if code_issues:
            report.findings.append(f"Found {len(code_issues)} files with long lines")
            report.findings.extend(code_issues[:5])  # Show first 5
        
        if missing_docstrings:
            report.findings.append(f"Found {len(missing_docstrings)} functions/classes without docstrings")
        
        if long_functions:
            report.findings.append(f"Found {len(long_functions)} functions longer than 50 lines")
            report.findings.extend(long_functions[:5])  # Show first 5
        
        # Recommendations
        report.recommendations.append("Add docstrings to all public functions and classes")
        report.recommendations.append("Follow PEP 8 style guide (max line length: 79-120 chars)")
        report.recommendations.append("Break down long functions into smaller, reusable components")
        report.recommendations.append("Use linters like flake8 or pylint for automated quality checks")
        report.recommendations.append("Implement code formatting with black or autopep8")
        
        report.metrics = {
            "long_line_files": len(code_issues),
            "missing_docstrings": len(missing_docstrings),
            "long_functions": len(long_functions)
        }
        
        report.priority = "medium"
        return report


class CryptoXSecurityAnalyst(Agent):
    """
    Security Analyst - CryptoX
    Responsibilities:
    - Conducts security audits
    - Identifies vulnerabilities and risks
    - Suggests security improvements
    """
    
    def __init__(self):
        super().__init__(
            name="CryptoX",
            role="Security Analyst",
            responsibilities=[
                "Conducts security audits on the software to identify vulnerabilities and risks.",
                "Suggests measures to improve the software's security and compliance.",
                "Monitors for any potential security threats and ensures secure development practices."
            ]
        )
    
    def analyze(self, project_path: str) -> AgentReport:
        """Analyze security aspects"""
        report = AgentReport(
            agent_name=self.name,
            role=self.role,
            timestamp=datetime.now().isoformat()
        )
        
        security_issues = []
        
        # Check for common security patterns
        py_files = list(Path(project_path).rglob("*.py"))
        
        for py_file in py_files:
            if '.git' in str(py_file) or '__pycache__' in str(py_file):
                continue
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Check for hardcoded secrets
                    if re.search(r'password\s*=\s*["\'][\w]+["\']', content, re.IGNORECASE):
                        security_issues.append(f"{py_file.name}: Potential hardcoded password")
                    
                    if re.search(r'api[_-]?key\s*=\s*["\'][\w]+["\']', content, re.IGNORECASE):
                        security_issues.append(f"{py_file.name}: Potential hardcoded API key")
                    
                    # Check for eval/exec usage
                    if 'eval(' in content or 'exec(' in content:
                        security_issues.append(f"{py_file.name}: Uses eval/exec (security risk)")
                    
                    # Check for pickle usage
                    if 'pickle.load' in content:
                        security_issues.append(f"{py_file.name}: Uses pickle.load (potential security risk)")
            except:
                continue
        
        # Check for .env file
        has_env = (Path(project_path) / ".env").exists()
        has_env_example = (Path(project_path) / ".env.example").exists()
        
        # Findings
        if security_issues:
            report.findings.append(f"⚠️ Found {len(security_issues)} potential security issues")
            report.findings.extend(security_issues)
        else:
            report.findings.append("✅ No obvious security issues detected in code scan")
        
        if has_env:
            report.findings.append("⚠️ .env file present - ensure it's in .gitignore")
        
        if not has_env_example:
            report.findings.append("💡 No .env.example file found")
        
        # Recommendations
        report.recommendations.append("Never commit secrets or API keys to version control")
        report.recommendations.append("Use environment variables for sensitive configuration")
        report.recommendations.append("Implement input validation for all user inputs")
        report.recommendations.append("Use parameterized queries to prevent SQL injection")
        report.recommendations.append("Add security headers for web applications")
        report.recommendations.append("Regularly update dependencies to patch vulnerabilities")
        report.recommendations.append("Consider using tools like bandit for Python security scanning")
        
        report.metrics = {
            "security_issues_found": len(security_issues),
            "has_env_file": has_env,
            "has_env_example": has_env_example
        }
        
        report.priority = "high" if security_issues else "medium"
        return report


class NOVAUIUXDesigner(Agent):
    """
    UI/UX Designer Assistant - NOVA
    Responsibilities:
    - Reviews user interface and experience
    - Suggests UI/UX improvements
    - Ensures design is technically feasible
    """
    
    def __init__(self):
        super().__init__(
            name="NOVA",
            role="UI/UX Designer Assistant",
            responsibilities=[
                "Reviews the user interface and user experience for intuitiveness and accessibility.",
                "Suggests improvements to UI/UX design to enhance user interaction and satisfaction.",
                "Collaborates with the lead developer to ensure the design is technically feasible."
            ]
        )
    
    def analyze(self, project_path: str) -> AgentReport:
        """Analyze UI/UX aspects"""
        report = AgentReport(
            agent_name=self.name,
            role=self.role,
            timestamp=datetime.now().isoformat()
        )
        
        # Look for UI-related files
        ui_indicators = {
            'gradio': 0,
            'streamlit': 0,
            'flask': 0,
            'fastapi': 0,
            'html': 0,
            'css': 0,
            'javascript': 0
        }
        
        # Check Python files for UI frameworks
        py_files = list(Path(project_path).rglob("*.py"))
        for py_file in py_files:
            if '.git' in str(py_file) or '__pycache__' in str(py_file):
                continue
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    for framework in ui_indicators:
                        if framework in content:
                            ui_indicators[framework] += 1
            except:
                continue
        
        # Check for web files
        html_files = list(Path(project_path).rglob("*.html"))
        css_files = list(Path(project_path).rglob("*.css"))
        js_files = list(Path(project_path).rglob("*.js"))
        
        # Findings
        ui_frameworks_found = [k for k, v in ui_indicators.items() if v > 0]
        
        if ui_frameworks_found:
            report.findings.append(f"UI frameworks detected: {', '.join(ui_frameworks_found)}")
        
        if html_files:
            report.findings.append(f"Found {len(html_files)} HTML files")
        if css_files:
            report.findings.append(f"Found {len(css_files)} CSS files")
        if js_files:
            report.findings.append(f"Found {len(js_files)} JavaScript files")
        
        if not any([ui_frameworks_found, html_files, css_files, js_files]):
            report.findings.append("No obvious UI components detected")
        
        # Recommendations
        report.recommendations.append("Ensure responsive design for different screen sizes")
        report.recommendations.append("Implement clear error messages and user feedback")
        report.recommendations.append("Add loading indicators for async operations")
        report.recommendations.append("Use consistent color scheme and typography")
        report.recommendations.append("Ensure accessibility (WCAG compliance)")
        report.recommendations.append("Add keyboard navigation support")
        report.recommendations.append("Implement dark mode option for better UX")
        
        report.metrics = {
            "ui_frameworks": ui_frameworks_found,
            "html_files": len(html_files),
            "css_files": len(css_files),
            "js_files": len(js_files)
        }
        
        report.priority = "medium"
        return report


class EchoPerformanceAnalyst(Agent):
    """
    Performance Analyst - Echo
    Responsibilities:
    - Analyzes software performance
    - Identifies bottlenecks
    - Proposes performance enhancements
    """
    
    def __init__(self):
        super().__init__(
            name="Echo",
            role="Performance Analyst",
            responsibilities=[
                "Analyzes the software's performance and identifies bottlenecks.",
                "Proposes performance enhancements to ensure scalability and speed.",
                "Optimizes algorithms and overall resource usage for maximum efficiency."
            ]
        )
    
    def analyze(self, project_path: str) -> AgentReport:
        """Analyze performance aspects"""
        report = AgentReport(
            agent_name=self.name,
            role=self.role,
            timestamp=datetime.now().isoformat()
        )
        
        performance_patterns = {
            'torch.cuda': 0,
            'multiprocessing': 0,
            'threading': 0,
            'asyncio': 0,
            '@lru_cache': 0,
            '@cache': 0,
            'numpy': 0
        }
        
        potential_issues = []
        
        py_files = list(Path(project_path).rglob("*.py"))
        for py_file in py_files:
            if '.git' in str(py_file) or '__pycache__' in str(py_file):
                continue
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Check for performance patterns
                    for pattern in performance_patterns:
                        if pattern in content:
                            performance_patterns[pattern] += 1
                    
                    # Check for potential issues
                    if 'for ' in content and 'append(' in content:
                        # Check for list comprehension opportunities
                        potential_issues.append(f"{py_file.name}: Consider list comprehensions instead of loops with append")
                    
                    # Check for nested loops (simplified check)
                    nested_for = content.count('for ') - content.count('for i in')
                    if nested_for > 3:
                        potential_issues.append(f"{py_file.name}: Multiple nested loops detected")
            except:
                continue
        
        # Findings
        optimizations_found = [k for k, v in performance_patterns.items() if v > 0]
        if optimizations_found:
            report.findings.append(f"Performance optimizations detected: {', '.join(optimizations_found)}")
        
        if potential_issues:
            report.findings.append(f"Found {len(potential_issues)} potential performance improvements")
            report.findings.extend(potential_issues[:3])
        
        # Recommendations
        report.recommendations.append("Profile code to identify actual bottlenecks")
        report.recommendations.append("Use GPU acceleration for ML operations when available")
        report.recommendations.append("Implement caching for expensive computations")
        report.recommendations.append("Consider batch processing for multiple items")
        report.recommendations.append("Use vectorized operations with NumPy instead of loops")
        report.recommendations.append("Optimize memory usage with generators for large datasets")
        report.recommendations.append("Implement lazy loading for resources")
        
        report.metrics = {
            "cuda_usage": performance_patterns['torch.cuda'] > 0,
            "parallel_processing": any([
                performance_patterns['multiprocessing'] > 0,
                performance_patterns['threading'] > 0,
                performance_patterns['asyncio'] > 0
            ]),
            "caching_used": any([
                performance_patterns['@lru_cache'] > 0,
                performance_patterns['@cache'] > 0
            ])
        }
        
        report.priority = "medium"
        return report


class SageDocumentationSpecialist(Agent):
    """
    Documentation Specialist - Sage
    Responsibilities:
    - Prepares comprehensive documentation
    - Tracks updates and modifications
    - Creates bilingual documentation
    """
    
    def __init__(self):
        super().__init__(
            name="Sage",
            role="Documentation Specialist",
            responsibilities=[
                "Prepares comprehensive documentation of the software architecture, features, and codebase.",
                "Keeps track of all updates and modifications and ensures clear version control.",
                "Prepares bilingual (English–Persian) documentation for internal and external stakeholders."
            ]
        )
    
    def analyze(self, project_path: str) -> AgentReport:
        """Analyze documentation quality"""
        report = AgentReport(
            agent_name=self.name,
            role=self.role,
            timestamp=datetime.now().isoformat()
        )
        
        # Check for documentation files
        doc_files = {
            'README.md': (Path(project_path) / "README.md").exists(),
            'CHANGELOG.md': (Path(project_path) / "CHANGELOG.md").exists(),
            'CONTRIBUTING.md': (Path(project_path) / "CONTRIBUTING.md").exists(),
            'LICENSE': any((Path(project_path) / f).exists() for f in ['LICENSE', 'LICENSE.md', 'LICENSE.txt']),
            'docs/': (Path(project_path) / "docs").exists(),
        }
        
        # Check for docstrings in code
        py_files = list(Path(project_path).rglob("*.py"))
        total_functions = 0
        documented_functions = 0
        
        for py_file in py_files:
            if '.git' in str(py_file) or '__pycache__' in str(py_file):
                continue
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            total_functions += 1
                            if ast.get_docstring(node):
                                documented_functions += 1
            except:
                continue
        
        # Findings
        for doc_file, exists in doc_files.items():
            if exists:
                report.findings.append(f"✅ {doc_file} present")
            else:
                report.findings.append(f"⚠️ {doc_file} missing")
        
        if total_functions > 0:
            doc_percentage = (documented_functions / total_functions) * 100
            report.findings.append(f"Documentation coverage: {doc_percentage:.1f}% ({documented_functions}/{total_functions} functions)")
        
        # Recommendations
        report.recommendations.append("Create comprehensive README with installation and usage instructions")
        report.recommendations.append("Add CHANGELOG to track version history")
        report.recommendations.append("Document API endpoints and function signatures")
        report.recommendations.append("Include code examples and tutorials")
        report.recommendations.append("Add inline comments for complex logic")
        report.recommendations.append("Create architecture diagrams for system overview")
        report.recommendations.append("Prepare bilingual documentation (English/Persian) as mentioned in goals")
        
        report.metrics = {
            "has_readme": doc_files['README.md'],
            "has_changelog": doc_files['CHANGELOG.md'],
            "has_license": doc_files['LICENSE'],
            "documentation_coverage_pct": (documented_functions / total_functions * 100) if total_functions > 0 else 0
        }
        
        report.priority = "medium"
        return report


class PulseDevOpsSpecialist(Agent):
    """
    DevOps Specialist - Pulse
    Responsibilities:
    - Ensures smooth integration and deployment
    - Manages CI/CD pipeline
    - Monitors release cycles
    """
    
    def __init__(self):
        super().__init__(
            name="Pulse",
            role="DevOps Specialist",
            responsibilities=[
                "Ensures the software can be smoothly integrated and deployed into different environments.",
                "Manages the CI/CD pipeline and automation of build and deployment processes.",
                "Monitors the integration processes and ensures stable release cycles."
            ]
        )
    
    def analyze(self, project_path: str) -> AgentReport:
        """Analyze DevOps and deployment aspects"""
        report = AgentReport(
            agent_name=self.name,
            role=self.role,
            timestamp=datetime.now().isoformat()
        )
        
        # Check for DevOps files
        devops_files = {
            '.github/workflows': (Path(project_path) / ".github" / "workflows").exists(),
            'Dockerfile': (Path(project_path) / "Dockerfile").exists(),
            'docker-compose.yml': (Path(project_path) / "docker-compose.yml").exists(),
            '.dockerignore': (Path(project_path) / ".dockerignore").exists(),
            '.gitignore': (Path(project_path) / ".gitignore").exists(),
            'Makefile': (Path(project_path) / "Makefile").exists(),
            'requirements.txt': (Path(project_path) / "requirements.txt").exists(),
        }
        
        # Check for CI/CD configurations
        ci_configs = list(Path(project_path).rglob(".github/workflows/*.yml"))
        
        # Findings
        for file_name, exists in devops_files.items():
            if exists:
                report.findings.append(f"✅ {file_name} present")
            else:
                report.findings.append(f"⚠️ {file_name} missing")
        
        if ci_configs:
            report.findings.append(f"Found {len(ci_configs)} CI/CD workflow(s)")
        else:
            report.findings.append("⚠️ No CI/CD workflows detected")
        
        # Recommendations
        report.recommendations.append("Set up GitHub Actions for automated testing and deployment")
        report.recommendations.append("Create Dockerfile for containerized deployment")
        report.recommendations.append("Implement automated testing in CI pipeline")
        report.recommendations.append("Add health check endpoints for monitoring")
        report.recommendations.append("Use environment-specific configuration files")
        report.recommendations.append("Implement automated dependency updates (Dependabot)")
        report.recommendations.append("Set up monitoring and logging for production")
        report.recommendations.append("Create deployment documentation and runbooks")
        
        report.metrics = {
            "has_ci_cd": len(ci_configs) > 0,
            "has_docker": devops_files['Dockerfile'],
            "has_gitignore": devops_files['.gitignore'],
            "ci_workflows_count": len(ci_configs)
        }
        
        report.priority = "medium"
        return report
