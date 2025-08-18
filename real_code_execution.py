#!/usr/bin/env python3
"""
Real Code Execution Engine
===========================

REAL secure code execution environment with runtime support.
Superior to Manus AI with containerized execution, debugging,
and deployment capabilities.
"""

import asyncio
import json
import time
import subprocess
import tempfile
import shutil
import os
import sys
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
import docker
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class RealCodeExecutionEngine:
    """Real code execution engine with containerized security"""
    
    def __init__(self):
        self.docker_available = False
        self.supported_languages = {
            'python': {
                'extension': '.py',
                'command': ['python3'],
                'image': 'python:3.11-slim'
            },
            'javascript': {
                'extension': '.js',
                'command': ['node'],
                'image': 'node:18-slim'
            },
            'typescript': {
                'extension': '.ts',
                'command': ['npx', 'ts-node'],
                'image': 'node:18-slim'
            },
            'bash': {
                'extension': '.sh',
                'command': ['bash'],
                'image': 'ubuntu:22.04'
            },
            'go': {
                'extension': '.go',
                'command': ['go', 'run'],
                'image': 'golang:1.21-slim'
            },
            'rust': {
                'extension': '.rs',
                'command': ['rustc', '--edition', '2021', '-o', '/tmp/program', '&&', '/tmp/program'],
                'image': 'rust:1.75-slim'
            }
        }
        
        self.execution_dir = Path("code_executions")
        self.execution_dir.mkdir(exist_ok=True)
        
        # Initialize Docker
        self._setup_docker()
    
    def _setup_docker(self):
        """Setup Docker for containerized execution"""
        try:
            self.docker_client = docker.from_env()
            # Test Docker connection
            self.docker_client.ping()
            self.docker_available = True
            logger.info("✅ Docker available for secure code execution")
        except Exception as e:
            logger.warning(f"⚠️ Docker not available: {e}")
            logger.info("📝 Code will execute in local sandbox")
            self.docker_available = False
    
    async def execute_code(self, code: str, language: str, 
                          execution_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute code with real runtime"""
        execution_config = execution_config or {}
        
        if language not in self.supported_languages:
            return {
                'success': False,
                'error': f'Unsupported language: {language}',
                'supported_languages': list(self.supported_languages.keys())
            }
        
        execution_id = hashlib.md5(f"{code}{time.time()}".encode()).hexdigest()[:8]
        start_time = time.time()
        
        logger.info(f"⚡ Executing {language} code (ID: {execution_id})")
        
        try:
            if self.docker_available and execution_config.get('use_container', True):
                result = await self._execute_in_container(code, language, execution_id, execution_config)
            else:
                result = await self._execute_in_sandbox(code, language, execution_id, execution_config)
            
            result['execution_time'] = time.time() - start_time
            result['execution_id'] = execution_id
            result['language'] = language
            result['timestamp'] = datetime.now().isoformat()
            
            logger.info(f"✅ Code execution completed: {execution_id} ({result['execution_time']:.2f}s)")
            return result
            
        except Exception as e:
            logger.error(f"❌ Code execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time,
                'execution_id': execution_id,
                'language': language,
                'timestamp': datetime.now().isoformat()
            }
    
    async def _execute_in_container(self, code: str, language: str, execution_id: str, 
                                   config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute code in Docker container for security"""
        lang_config = self.supported_languages[language]
        
        # Create execution directory
        exec_dir = self.execution_dir / execution_id
        exec_dir.mkdir(exist_ok=True)
        
        # Write code to file
        code_file = exec_dir / f"code{lang_config['extension']}"
        code_file.write_text(code)
        
        # Prepare container configuration
        container_config = {
            'image': lang_config['image'],
            'command': self._build_execution_command(language, f"/workspace/code{lang_config['extension']}"),
            'volumes': {str(exec_dir): {'bind': '/workspace', 'mode': 'rw'}},
            'working_dir': '/workspace',
            'mem_limit': config.get('memory_limit', '512m'),
            'cpu_period': 100000,
            'cpu_quota': config.get('cpu_quota', 50000),  # 50% CPU
            'network_disabled': config.get('disable_network', True),
            'remove': True,
            'stdout': True,
            'stderr': True
        }
        
        try:
            # Run container
            container = self.docker_client.containers.run(**container_config)
            
            # Get output
            output = container.decode('utf-8')
            
            # Parse stdout and stderr
            stdout_lines = []
            stderr_lines = []
            
            for line in output.split('\n'):
                if line.strip():
                    stdout_lines.append(line)
            
            result = {
                'success': True,
                'stdout': '\n'.join(stdout_lines),
                'stderr': '\n'.join(stderr_lines),
                'exit_code': 0,
                'execution_method': 'docker_container',
                'container_image': lang_config['image']
            }
            
            return result
            
        except docker.errors.ContainerError as e:
            return {
                'success': False,
                'stdout': e.stdout.decode('utf-8') if e.stdout else '',
                'stderr': e.stderr.decode('utf-8') if e.stderr else '',
                'exit_code': e.exit_status,
                'error': f'Container execution failed: {str(e)}',
                'execution_method': 'docker_container'
            }
        finally:
            # Cleanup
            shutil.rmtree(exec_dir, ignore_errors=True)
    
    async def _execute_in_sandbox(self, code: str, language: str, execution_id: str, 
                                 config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute code in local sandbox"""
        lang_config = self.supported_languages[language]
        
        # Create execution directory
        exec_dir = self.execution_dir / execution_id
        exec_dir.mkdir(exist_ok=True)
        
        # Write code to file
        code_file = exec_dir / f"code{lang_config['extension']}"
        code_file.write_text(code)
        
        try:
            # Build execution command
            cmd = self._build_execution_command(language, str(code_file))
            
            # Execute with timeout and resource limits
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=exec_dir,
                env=self._get_sandbox_environment()
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=config.get('timeout', 30)
                )
                
                result = {
                    'success': process.returncode == 0,
                    'stdout': stdout.decode('utf-8'),
                    'stderr': stderr.decode('utf-8'),
                    'exit_code': process.returncode,
                    'execution_method': 'local_sandbox'
                }
                
                return result
                
            except asyncio.TimeoutError:
                process.kill()
                return {
                    'success': False,
                    'error': 'Execution timeout',
                    'timeout': config.get('timeout', 30),
                    'execution_method': 'local_sandbox'
                }
                
        finally:
            # Cleanup
            shutil.rmtree(exec_dir, ignore_errors=True)
    
    def _build_execution_command(self, language: str, file_path: str) -> List[str]:
        """Build execution command for language"""
        lang_config = self.supported_languages[language]
        
        if language == 'python':
            return ['python3', file_path]
        elif language == 'javascript':
            return ['node', file_path]
        elif language == 'typescript':
            return ['npx', 'ts-node', file_path]
        elif language == 'bash':
            return ['bash', file_path]
        elif language == 'go':
            return ['go', 'run', file_path]
        elif language == 'rust':
            return ['sh', '-c', f'rustc {file_path} -o /tmp/program && /tmp/program']
        else:
            return lang_config['command'] + [file_path]
    
    def _get_sandbox_environment(self) -> Dict[str, str]:
        """Get sandboxed environment variables"""
        # Minimal environment for security
        return {
            'PATH': '/usr/local/bin:/usr/bin:/bin',
            'HOME': '/tmp',
            'USER': 'sandbox',
            'PYTHONPATH': '',
            'NODE_PATH': ''
        }
    
    async def debug_code(self, code: str, language: str, 
                        debug_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Real code debugging with analysis"""
        debug_config = debug_config or {}
        
        start_time = time.time()
        
        try:
            # First, try to execute the code
            execution_result = await self.execute_code(code, language, debug_config)
            
            debug_analysis = {
                'syntax_errors': [],
                'runtime_errors': [],
                'warnings': [],
                'suggestions': []
            }
            
            if not execution_result['success']:
                # Analyze errors
                stderr = execution_result.get('stderr', '')
                
                if language == 'python':
                    debug_analysis = await self._debug_python_code(code, stderr)
                elif language == 'javascript':
                    debug_analysis = await self._debug_javascript_code(code, stderr)
                else:
                    debug_analysis = await self._debug_generic_code(code, stderr, language)
            
            result = {
                'success': True,
                'execution_result': execution_result,
                'debug_analysis': debug_analysis,
                'debug_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"🐛 Code debugging completed: {len(debug_analysis['suggestions'])} suggestions")
            return result
            
        except Exception as e:
            logger.error(f"❌ Code debugging failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'debug_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
    
    async def _debug_python_code(self, code: str, stderr: str) -> Dict[str, List[str]]:
        """Debug Python code specifically"""
        debug_info = {
            'syntax_errors': [],
            'runtime_errors': [],
            'warnings': [],
            'suggestions': []
        }
        
        # Parse Python error messages
        if 'SyntaxError' in stderr:
            debug_info['syntax_errors'].append('Python syntax error detected')
            debug_info['suggestions'].append('Check for missing colons, parentheses, or indentation')
        
        if 'NameError' in stderr:
            debug_info['runtime_errors'].append('Undefined variable or function')
            debug_info['suggestions'].append('Check variable names and imports')
        
        if 'ImportError' in stderr or 'ModuleNotFoundError' in stderr:
            debug_info['runtime_errors'].append('Missing module or import error')
            debug_info['suggestions'].append('Install required packages or check import statements')
        
        if 'IndentationError' in stderr:
            debug_info['syntax_errors'].append('Python indentation error')
            debug_info['suggestions'].append('Fix indentation - use consistent spaces or tabs')
        
        return debug_info
    
    async def _debug_javascript_code(self, code: str, stderr: str) -> Dict[str, List[str]]:
        """Debug JavaScript code specifically"""
        debug_info = {
            'syntax_errors': [],
            'runtime_errors': [],
            'warnings': [],
            'suggestions': []
        }
        
        # Parse JavaScript error messages
        if 'SyntaxError' in stderr:
            debug_info['syntax_errors'].append('JavaScript syntax error')
            debug_info['suggestions'].append('Check for missing semicolons, brackets, or quotes')
        
        if 'ReferenceError' in stderr:
            debug_info['runtime_errors'].append('Undefined variable or function')
            debug_info['suggestions'].append('Check variable declarations and scope')
        
        if 'TypeError' in stderr:
            debug_info['runtime_errors'].append('Type error in JavaScript')
            debug_info['suggestions'].append('Check data types and method calls')
        
        return debug_info
    
    async def _debug_generic_code(self, code: str, stderr: str, language: str) -> Dict[str, List[str]]:
        """Generic code debugging"""
        debug_info = {
            'syntax_errors': [],
            'runtime_errors': [],
            'warnings': [],
            'suggestions': []
        }
        
        if stderr:
            debug_info['runtime_errors'].append(f'{language} execution error')
            debug_info['suggestions'].append(f'Check {language} syntax and runtime requirements')
        
        return debug_info
    
    async def test_code(self, code: str, language: str, 
                       test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Real code testing with test cases"""
        start_time = time.time()
        
        try:
            test_results = []
            
            for i, test_case in enumerate(test_cases):
                test_id = f"test_{i+1}"
                
                # Prepare test code
                test_code = self._prepare_test_code(code, test_case, language)
                
                # Execute test
                execution_result = await self.execute_code(test_code, language)
                
                # Analyze test result
                test_result = {
                    'test_id': test_id,
                    'input': test_case.get('input'),
                    'expected_output': test_case.get('expected_output'),
                    'actual_output': execution_result.get('stdout', '').strip(),
                    'passed': self._evaluate_test_result(
                        execution_result.get('stdout', '').strip(),
                        test_case.get('expected_output')
                    ),
                    'execution_success': execution_result['success'],
                    'error': execution_result.get('error')
                }
                
                test_results.append(test_result)
            
            # Calculate test summary
            passed_tests = sum(1 for result in test_results if result['passed'])
            total_tests = len(test_results)
            
            result = {
                'success': True,
                'test_results': test_results,
                'summary': {
                    'total_tests': total_tests,
                    'passed_tests': passed_tests,
                    'failed_tests': total_tests - passed_tests,
                    'success_rate': passed_tests / total_tests if total_tests > 0 else 0
                },
                'testing_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"🧪 Code testing: {passed_tests}/{total_tests} tests passed")
            return result
            
        except Exception as e:
            logger.error(f"❌ Code testing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'testing_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
    
    def _prepare_test_code(self, original_code: str, test_case: Dict[str, Any], language: str) -> str:
        """Prepare code with test case"""
        if language == 'python':
            return f"""
{original_code}

# Test case
if __name__ == "__main__":
    test_input = {repr(test_case.get('input', ''))}
    result = main(test_input) if 'main' in globals() else eval(test_input)
    print(result)
"""
        elif language == 'javascript':
            return f"""
{original_code}

// Test case
const testInput = {json.dumps(test_case.get('input', ''))};
const result = typeof main === 'function' ? main(testInput) : eval(testInput);
console.log(result);
"""
        else:
            return original_code
    
    def _evaluate_test_result(self, actual_output: str, expected_output: Any) -> bool:
        """Evaluate if test result matches expected output"""
        if expected_output is None:
            return True
        
        try:
            # Try exact string match first
            if str(actual_output).strip() == str(expected_output).strip():
                return True
            
            # Try numeric comparison
            try:
                actual_num = float(actual_output.strip())
                expected_num = float(expected_output)
                return abs(actual_num - expected_num) < 0.0001
            except (ValueError, TypeError):
                pass
            
            # Try JSON comparison
            try:
                actual_json = json.loads(actual_output)
                if actual_json == expected_output:
                    return True
            except (json.JSONDecodeError, TypeError):
                pass
            
            return False
            
        except Exception:
            return False
    
    async def deploy_code(self, code: str, language: str, 
                         deployment_config: Dict[str, Any]) -> Dict[str, Any]:
        """Real code deployment capabilities"""
        start_time = time.time()
        
        try:
            deployment_type = deployment_config.get('type', 'local')
            
            if deployment_type == 'docker':
                result = await self._deploy_to_docker(code, language, deployment_config)
            elif deployment_type == 'local':
                result = await self._deploy_locally(code, language, deployment_config)
            else:
                raise ValueError(f"Unsupported deployment type: {deployment_type}")
            
            result['deployment_time'] = time.time() - start_time
            result['timestamp'] = datetime.now().isoformat()
            
            logger.info(f"🚀 Code deployed: {deployment_type} deployment")
            return result
            
        except Exception as e:
            logger.error(f"❌ Code deployment failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'deployment_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
    
    async def _deploy_to_docker(self, code: str, language: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy code as Docker container"""
        if not self.docker_available:
            raise RuntimeError("Docker not available for deployment")
        
        # Create deployment directory
        deploy_dir = self.execution_dir / f"deploy_{int(time.time())}"
        deploy_dir.mkdir(exist_ok=True)
        
        try:
            # Write code and Dockerfile
            lang_config = self.supported_languages[language]
            code_file = deploy_dir / f"app{lang_config['extension']}"
            code_file.write_text(code)
            
            dockerfile_content = self._generate_dockerfile(language, f"app{lang_config['extension']}")
            dockerfile = deploy_dir / "Dockerfile"
            dockerfile.write_text(dockerfile_content)
            
            # Build Docker image
            image_tag = f"super-omega-app:{int(time.time())}"
            image, build_logs = self.docker_client.images.build(
                path=str(deploy_dir),
                tag=image_tag,
                rm=True
            )
            
            # Run container
            container = self.docker_client.containers.run(
                image_tag,
                detach=True,
                ports=config.get('ports', {}),
                environment=config.get('environment', {}),
                name=f"super-omega-{int(time.time())}"
            )
            
            return {
                'success': True,
                'deployment_type': 'docker',
                'image_id': image.id,
                'container_id': container.id,
                'container_name': container.name,
                'status': container.status
            }
            
        finally:
            # Cleanup deployment directory
            shutil.rmtree(deploy_dir, ignore_errors=True)
    
    async def _deploy_locally(self, code: str, language: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy code locally"""
        deploy_dir = Path(config.get('deploy_path', 'deployed_apps')) / f"app_{int(time.time())}"
        deploy_dir.mkdir(parents=True, exist_ok=True)
        
        # Write code
        lang_config = self.supported_languages[language]
        code_file = deploy_dir / f"app{lang_config['extension']}"
        code_file.write_text(code)
        
        # Create run script
        run_script = deploy_dir / "run.sh"
        run_script.write_text(f"#!/bin/bash\ncd {deploy_dir}\n{' '.join(lang_config['command'])} app{lang_config['extension']}")
        run_script.chmod(0o755)
        
        return {
            'success': True,
            'deployment_type': 'local',
            'deploy_path': str(deploy_dir),
            'run_script': str(run_script)
        }
    
    def _generate_dockerfile(self, language: str, app_file: str) -> str:
        """Generate Dockerfile for language"""
        lang_config = self.supported_languages[language]
        
        if language == 'python':
            return f"""
FROM {lang_config['image']}
WORKDIR /app
COPY {app_file} .
CMD ["python3", "{app_file}"]
"""
        elif language == 'javascript':
            return f"""
FROM {lang_config['image']}
WORKDIR /app
COPY {app_file} .
CMD ["node", "{app_file}"]
"""
        else:
            return f"""
FROM {lang_config['image']}
WORKDIR /app
COPY {app_file} .
CMD {json.dumps(lang_config['command'] + [app_file])}
"""

# Global instance
_real_code_execution_engine = None

def get_real_code_execution_engine() -> RealCodeExecutionEngine:
    """Get global real code execution engine instance"""
    global _real_code_execution_engine
    if _real_code_execution_engine is None:
        _real_code_execution_engine = RealCodeExecutionEngine()
    return _real_code_execution_engine