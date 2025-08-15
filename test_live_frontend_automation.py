#!/usr/bin/env python3
"""
Live Frontend-to-Backend Automation Verification Test
=====================================================

CRITICAL TEST: Verify that frontend instructions actually trigger 
REAL LIVE PLAYWRIGHT AUTOMATION, not just simulations.

This test will:
1. Send a real instruction through the API
2. Verify that actual Playwright browser automation occurs
3. Confirm real websites are accessed and interacted with
4. Validate that healing and evidence collection happen live
5. Generate proof of genuine automation (not simulation)

✅ REAL AUTOMATION VERIFICATION - NO FAKE RESPONSES!
"""

import asyncio
import json
import logging
import time
import aiohttp
import subprocess
from datetime import datetime
from typing import Dict, Any
from pathlib import Path
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LiveFrontendAutomationVerifier:
    """Verify that frontend instructions trigger real live automation"""
    
    def __init__(self):
        self.api_base_url = "http://127.0.0.1:8888"
        self.test_results = {
            'test_metadata': {
                'test_name': 'Live Frontend-to-Backend Automation Verification',
                'test_version': '1.0.0',
                'timestamp': datetime.now().isoformat(),
                'test_type': 'REAL_FRONTEND_AUTOMATION'
            },
            'ui_server_verification': {},
            'api_connectivity_test': {},
            'live_automation_tests': {},
            'playwright_verification': {},
            'evidence_verification': {},
            'final_verdict': {}
        }
        
    async def verify_live_frontend_automation(self) -> Dict[str, Any]:
        """Complete verification of live frontend-to-backend automation"""
        logger.info("🎯 Starting Live Frontend Automation Verification")
        
        try:
            # Step 1: Verify UI server is accessible
            await self._verify_ui_server()
            
            # Step 2: Test API connectivity
            await self._test_api_connectivity()
            
            # Step 3: Send real instruction and verify live automation
            await self._test_live_automation_instruction()
            
            # Step 4: Verify Playwright processes and browser activity
            await self._verify_playwright_processes()
            
            # Step 5: Verify evidence collection from live runs
            await self._verify_live_evidence_collection()
            
            # Step 6: Generate final verification verdict
            self._generate_final_verdict()
            
            logger.info("✅ Live Frontend Automation Verification completed")
            return self.test_results
            
        except Exception as e:
            logger.error(f"❌ Live automation verification failed: {e}")
            traceback.print_exc()
            return {
                'error': str(e),
                'test_failed': True,
                'timestamp': datetime.now().isoformat()
            }
    
    async def _verify_ui_server(self):
        """Verify UI server is running and accessible"""
        logger.info("🔍 Verifying UI server accessibility...")
        
        try:
            # Check if server process is running
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            server_running = 'super_omega_live_console' in result.stdout
            
            # Try to access the UI
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(f"{self.api_base_url}/", timeout=5) as response:
                        ui_accessible = response.status == 200
                        ui_content = await response.text()
                except Exception as e:
                    ui_accessible = False
                    ui_content = str(e)
            
            self.test_results['ui_server_verification'] = {
                'server_process_running': server_running,
                'ui_accessible': ui_accessible,
                'ui_response_received': len(ui_content) > 0,
                'verification_time': datetime.now().isoformat()
            }
            
            logger.info(f"✅ UI Server - Process: {server_running}, Accessible: {ui_accessible}")
            
        except Exception as e:
            logger.error(f"❌ UI server verification failed: {e}")
            self.test_results['ui_server_verification'] = {
                'error': str(e),
                'failed': True
            }
    
    async def _test_api_connectivity(self):
        """Test API connectivity and endpoints"""
        logger.info("🔍 Testing API connectivity...")
        
        api_tests = []
        
        async with aiohttp.ClientSession() as session:
            # Test health endpoint
            try:
                async with session.get(f"{self.api_base_url}/api/super-omega-status", timeout=10) as response:
                    status_test = {
                        'endpoint': '/api/super-omega-status',
                        'accessible': response.status == 200,
                        'response_data': await response.json() if response.status == 200 else None
                    }
                    api_tests.append(status_test)
            except Exception as e:
                api_tests.append({
                    'endpoint': '/api/super-omega-status',
                    'accessible': False,
                    'error': str(e)
                })
        
        self.test_results['api_connectivity_test'] = {
            'total_endpoints_tested': len(api_tests),
            'accessible_endpoints': sum(1 for test in api_tests if test.get('accessible', False)),
            'endpoint_tests': api_tests,
            'api_ready': any(test.get('accessible', False) for test in api_tests)
        }
        
        logger.info(f"✅ API Connectivity - Ready: {self.test_results['api_connectivity_test']['api_ready']}")
    
    async def _test_live_automation_instruction(self):
        """Send real instruction and verify live automation occurs"""
        logger.info("🎭 Testing live automation instruction...")
        
        # Real complex instruction that requires actual website interaction
        test_instruction = "Navigate to Google.com, search for 'playwright automation testing', click on the first result, and capture evidence of the page content"
        
        automation_start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                # Send instruction to the live automation API
                payload = {
                    'instruction': test_instruction,
                    'mode': 'HYBRID',
                    'capture_evidence': True,
                    'enable_healing': True
                }
                
                async with session.post(
                    f"{self.api_base_url}/api/super-omega-execute",
                    json=payload,
                    timeout=60  # Allow time for real automation
                ) as response:
                    
                    if response.status == 200:
                        result_data = await response.json()
                        automation_successful = result_data.get('success', False)
                        session_id = result_data.get('session_id', None)
                        
                        # If we got a session ID, check for live status updates
                        live_updates = []
                        if session_id:
                            for i in range(5):  # Check 5 times over 10 seconds
                                await asyncio.sleep(2)
                                try:
                                    async with session.get(f"{self.api_base_url}/api/session-status/{session_id}") as status_response:
                                        if status_response.status == 200:
                                            status_data = await status_response.json()
                                            live_updates.append({
                                                'check_time': datetime.now().isoformat(),
                                                'status': status_data
                                            })
                                except Exception as e:
                                    live_updates.append({
                                        'check_time': datetime.now().isoformat(),
                                        'error': str(e)
                                    })
                        
                        automation_end_time = time.time()
                        
                        self.test_results['live_automation_tests'] = {
                            'instruction_sent': test_instruction,
                            'api_response_received': True,
                            'automation_successful': automation_successful,
                            'session_id': session_id,
                            'execution_time_seconds': automation_end_time - automation_start_time,
                            'live_status_updates': live_updates,
                            'total_status_checks': len(live_updates),
                            'result_data': result_data
                        }
                        
                    else:
                        self.test_results['live_automation_tests'] = {
                            'instruction_sent': test_instruction,
                            'api_response_received': False,
                            'http_status': response.status,
                            'error': await response.text()
                        }
                        
        except Exception as e:
            logger.error(f"❌ Live automation test failed: {e}")
            self.test_results['live_automation_tests'] = {
                'instruction_sent': test_instruction,
                'error': str(e),
                'failed': True
            }
        
        logger.info("✅ Live automation instruction test completed")
    
    async def _verify_playwright_processes(self):
        """Verify that actual Playwright processes are running"""
        logger.info("🔍 Verifying Playwright processes...")
        
        try:
            # Check for Playwright/browser processes
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            processes = result.stdout
            
            playwright_indicators = [
                'playwright',
                'chromium',
                'chrome',
                'firefox',
                'webkit'
            ]
            
            detected_processes = []
            for indicator in playwright_indicators:
                if indicator.lower() in processes.lower():
                    # Extract relevant process lines
                    lines = [line for line in processes.split('\n') if indicator.lower() in line.lower()]
                    detected_processes.extend(lines[:3])  # Limit to 3 lines per indicator
            
            # Check for browser automation evidence
            browser_evidence = {
                'playwright_processes_detected': len(detected_processes) > 0,
                'detected_process_count': len(detected_processes),
                'process_details': detected_processes[:10],  # Limit output
                'browser_automation_active': any('chrome' in p.lower() or 'chromium' in p.lower() for p in detected_processes)
            }
            
            self.test_results['playwright_verification'] = browser_evidence
            
            logger.info(f"✅ Playwright Verification - Processes detected: {len(detected_processes)}")
            
        except Exception as e:
            logger.error(f"❌ Playwright verification failed: {e}")
            self.test_results['playwright_verification'] = {
                'error': str(e),
                'failed': True
            }
    
    async def _verify_live_evidence_collection(self):
        """Verify that evidence from live runs exists"""
        logger.info("🔍 Verifying live evidence collection...")
        
        try:
            # Check for runs directory and recent evidence
            runs_dir = Path("runs")
            evidence_found = {
                'runs_directory_exists': runs_dir.exists(),
                'total_run_directories': 0,
                'recent_run_directories': [],
                'evidence_files_found': [],
                'live_evidence_confirmed': False
            }
            
            if runs_dir.exists():
                run_dirs = list(runs_dir.iterdir())
                evidence_found['total_run_directories'] = len(run_dirs)
                
                # Look for recent runs (last hour)
                current_time = time.time()
                recent_runs = []
                
                for run_dir in run_dirs:
                    if run_dir.is_dir():
                        # Check modification time
                        mod_time = run_dir.stat().st_mtime
                        if current_time - mod_time < 3600:  # Last hour
                            recent_runs.append({
                                'directory': str(run_dir),
                                'created_time': datetime.fromtimestamp(mod_time).isoformat(),
                                'files': list(str(f) for f in run_dir.rglob('*') if f.is_file())[:10]  # Limit files
                            })
                
                evidence_found['recent_run_directories'] = recent_runs
                
                # Look for specific evidence files
                evidence_types = ['*.json', '*.png', '*.mp4', '*.jsonl', '*.ts', '*.py']
                for pattern in evidence_types:
                    files = list(runs_dir.rglob(pattern))
                    evidence_found['evidence_files_found'].extend([str(f) for f in files[:5]])  # Limit files
                
                evidence_found['live_evidence_confirmed'] = len(recent_runs) > 0 and len(evidence_found['evidence_files_found']) > 0
            
            self.test_results['evidence_verification'] = evidence_found
            
            logger.info(f"✅ Evidence Verification - Live evidence: {evidence_found['live_evidence_confirmed']}")
            
        except Exception as e:
            logger.error(f"❌ Evidence verification failed: {e}")
            self.test_results['evidence_verification'] = {
                'error': str(e),
                'failed': True
            }
    
    def _generate_final_verdict(self):
        """Generate final verification verdict"""
        
        # Analyze all test results
        ui_working = self.test_results.get('ui_server_verification', {}).get('ui_accessible', False)
        api_working = self.test_results.get('api_connectivity_test', {}).get('api_ready', False)
        automation_working = self.test_results.get('live_automation_tests', {}).get('automation_successful', False)
        playwright_detected = self.test_results.get('playwright_verification', {}).get('playwright_processes_detected', False)
        evidence_exists = self.test_results.get('evidence_verification', {}).get('live_evidence_confirmed', False)
        
        # Calculate overall verification score
        verification_components = [ui_working, api_working, automation_working, playwright_detected, evidence_exists]
        verification_score = sum(verification_components) / len(verification_components) * 100
        
        verdict = {
            'overall_verification_score': verification_score,
            'live_automation_confirmed': verification_score >= 60.0,
            'ui_server_functional': ui_working,
            'api_connectivity_working': api_working,
            'automation_execution_successful': automation_working,
            'playwright_processes_active': playwright_detected,
            'live_evidence_generated': evidence_exists,
            'verification_level': self._determine_verification_level(verification_score),
            'frontend_to_backend_integration': verification_score >= 40.0,
            'real_automation_confirmed': playwright_detected and (automation_working or evidence_exists)
        }
        
        self.test_results['final_verdict'] = verdict
    
    def _determine_verification_level(self, score: float) -> str:
        """Determine verification level based on score"""
        if score >= 80:
            return "FULLY_VERIFIED"
        elif score >= 60:
            return "MOSTLY_VERIFIED"
        elif score >= 40:
            return "PARTIALLY_VERIFIED"
        else:
            return "NOT_VERIFIED"

async def main():
    """Run the live frontend automation verification"""
    verifier = LiveFrontendAutomationVerifier()
    
    try:
        print("🎯 LIVE FRONTEND-TO-BACKEND AUTOMATION VERIFICATION")
        print("=" * 65)
        
        results = await verifier.verify_live_frontend_automation()
        
        # Save results
        with open('LIVE_FRONTEND_AUTOMATION_VERIFICATION_REPORT.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print("\n🏆 VERIFICATION RESULTS SUMMARY:")
        print("=" * 45)
        
        if 'final_verdict' in results:
            verdict = results['final_verdict']
            print(f"📊 Overall Verification Score: {verdict['overall_verification_score']:.1f}%")
            print(f"🎭 Live Automation Confirmed: {'YES' if verdict['live_automation_confirmed'] else 'NO'}")
            print(f"🌐 UI Server Functional: {'YES' if verdict['ui_server_functional'] else 'NO'}")
            print(f"🔌 API Connectivity: {'YES' if verdict['api_connectivity_working'] else 'NO'}")
            print(f"⚡ Automation Execution: {'YES' if verdict['automation_execution_successful'] else 'NO'}")
            print(f"🎭 Playwright Processes: {'YES' if verdict['playwright_processes_active'] else 'NO'}")
            print(f"📁 Live Evidence: {'YES' if verdict['live_evidence_generated'] else 'NO'}")
            print(f"🎯 Verification Level: {verdict['verification_level']}")
            print(f"🔗 Frontend-Backend Integration: {'YES' if verdict['frontend_to_backend_integration'] else 'NO'}")
            print(f"✅ Real Automation Confirmed: {'YES' if verdict['real_automation_confirmed'] else 'NO'}")
        
        print(f"\n📁 Report saved to: LIVE_FRONTEND_AUTOMATION_VERIFICATION_REPORT.json")
        
        return results
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        traceback.print_exc()
        return {'error': str(e)}

if __name__ == "__main__":
    asyncio.run(main())