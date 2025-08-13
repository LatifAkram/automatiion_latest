#!/usr/bin/env python3
"""
Example Usage of the Autonomous Multi-Agent Automation Platform
==============================================================

This script demonstrates how to use the automation platform to execute workflows.
"""

import asyncio
import requests
import json
from datetime import datetime


async def main():
    """Demonstrate the automation platform capabilities."""
    
    print("🚀 Autonomous Multi-Agent Automation Platform Demo")
    print("=" * 60)
    
    # API base URL
    base_url = "http://localhost:8000"
    
    try:
        # 1. Check system health
        print("\n1. Checking system health...")
        health_response = requests.get(f"{base_url}/health")
        if health_response.status_code == 200:
            print("✅ System is healthy")
        else:
            print("❌ System health check failed")
            return
            
        # 2. Get system information
        print("\n2. Getting system information...")
        system_response = requests.get(f"{base_url}/system/info")
        if system_response.status_code == 200:
            system_info = system_response.json()
            print(f"✅ Platform version: {system_info.get('version')}")
            print(f"✅ Environment: {system_info.get('environment')}")
            print(f"✅ AI Providers: {system_info.get('ai_providers')}")
        else:
            print("❌ Failed to get system information")
            
        # 3. Create a sample workflow
        print("\n3. Creating a sample workflow...")
        workflow_request = {
            "name": "E-commerce Price Comparison Demo",
            "description": "Compare laptop prices across multiple e-commerce sites",
            "domain": "ecommerce",
            "parameters": {
                "product": "laptop",
                "budget": 1500,
                "sites": ["amazon", "bestbuy", "newegg"],
                "features": ["gaming", "business", "student"]
            },
            "tags": ["demo", "ecommerce", "price-comparison"]
        }
        
        workflow_response = requests.post(f"{base_url}/workflows", json=workflow_request)
        if workflow_response.status_code == 200:
            workflow_data = workflow_response.json()
            workflow_id = workflow_data["workflow_id"]
            print(f"✅ Workflow created: {workflow_id}")
            print(f"✅ Status: {workflow_data['status']}")
            print(f"✅ Message: {workflow_data['message']}")
        else:
            print("❌ Failed to create workflow")
            return
            
        # 4. Check workflow status
        print("\n4. Checking workflow status...")
        status_response = requests.get(f"{base_url}/workflows/{workflow_id}")
        if status_response.status_code == 200:
            status_data = status_response.json()
            print(f"✅ Workflow status: {status_data.get('status')}")
            print(f"✅ Created at: {status_data.get('created_at')}")
        else:
            print("❌ Failed to get workflow status")
            
        # 5. Chat with the AI agent
        print("\n5. Chatting with AI agent...")
        chat_request = {
            "message": "What's the status of my e-commerce price comparison workflow?",
            "session_id": "demo_session",
            "context": {
                "workflow_id": workflow_id,
                "user_type": "demo"
            }
        }
        
        chat_response = requests.post(f"{base_url}/chat", json=chat_request)
        if chat_response.status_code == 200:
            chat_data = chat_response.json()
            print(f"✅ AI Response: {chat_data['response'][:100]}...")
            print(f"✅ Session ID: {chat_data['session_id']}")
        else:
            print("❌ Failed to chat with AI agent")
            
        # 6. Get performance metrics
        print("\n6. Getting performance metrics...")
        metrics_response = requests.get(f"{base_url}/analytics/performance")
        if metrics_response.status_code == 200:
            metrics_data = metrics_response.json()
            print(f"✅ Active workflows: {metrics_data.get('active_workflows')}")
            print(f"✅ Execution agents: {metrics_data.get('execution_agents')}")
        else:
            print("❌ Failed to get performance metrics")
            
        # 7. Get agent status
        print("\n7. Getting agent status...")
        agents_response = requests.get(f"{base_url}/analytics/agents")
        if agents_response.status_code == 200:
            agents_data = agents_response.json()
            print(f"✅ Total agents: {agents_data.get('total_agents')}")
            for agent in agents_data.get('agents', []):
                print(f"   - {agent.get('type')}: {agent.get('agent_id')} ({agent.get('status', 'active')})")
        else:
            print("❌ Failed to get agent status")
            
        # 8. List workflows
        print("\n8. Listing workflows...")
        workflows_response = requests.get(f"{base_url}/workflows")
        if workflows_response.status_code == 200:
            workflows_data = workflows_response.json()
            print(f"✅ Total workflows: {workflows_data.get('total')}")
            for workflow in workflows_data.get('workflows', []):
                print(f"   - {workflow.get('name')}: {workflow.get('status')}")
        else:
            print("❌ Failed to list workflows")
            
        print("\n🎉 Demo completed successfully!")
        print("\nNext steps:")
        print("1. Check the API documentation at http://localhost:8000/docs")
        print("2. Explore the workflow execution logs")
        print("3. Try creating more complex workflows")
        print("4. Experiment with different AI providers")
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to the automation platform")
        print("Make sure the platform is running with: python main.py")
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")


if __name__ == "__main__":
    asyncio.run(main())