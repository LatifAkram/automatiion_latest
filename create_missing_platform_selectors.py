#!/usr/bin/env python3
"""
CREATE MISSING PLATFORM SELECTORS
=================================
Generate selector databases for missing platforms to achieve 100% coverage
"""

import sqlite3
import json
from datetime import datetime

def create_google_selectors():
    """Create comprehensive Google selectors"""
    selectors = []
    
    # Google search selectors
    google_selectors = [
        # Search input
        {'selector': 'input[name="q"]', 'type': 'css', 'action': 'type', 'confidence': 0.95},
        {'selector': 'textarea[name="q"]', 'type': 'css', 'action': 'type', 'confidence': 0.90},
        {'selector': '[title="Search"]', 'type': 'css', 'action': 'type', 'confidence': 0.85},
        {'selector': '#APjFqb', 'type': 'css', 'action': 'type', 'confidence': 0.80},
        
        # Search buttons
        {'selector': 'input[name="btnK"]', 'type': 'css', 'action': 'click', 'confidence': 0.95},
        {'selector': '[aria-label="Google Search"]', 'type': 'css', 'action': 'click', 'confidence': 0.90},
        {'selector': '.gNO89b', 'type': 'css', 'action': 'click', 'confidence': 0.85},
        
        # Search results
        {'selector': '.g', 'type': 'css', 'action': 'click', 'confidence': 0.90},
        {'selector': '.yuRUbf a', 'type': 'css', 'action': 'click', 'confidence': 0.95},
        {'selector': 'h3', 'type': 'css', 'action': 'click', 'confidence': 0.85},
        
        # Navigation
        {'selector': '.gb_d', 'type': 'css', 'action': 'click', 'confidence': 0.80},
        {'selector': '[role="button"]', 'type': 'css', 'action': 'click', 'confidence': 0.75}
    ]
    
    for i, sel in enumerate(google_selectors):
        selectors.append({
            'id': i + 1,
            'platform': 'google',
            'category': 'search',
            'module': 'main',
            'selector_type': sel['type'],
            'action_type': sel['action'],
            'complexity': 'simple',
            'selector': sel['selector'],
            'fallback_selectors': json.dumps([]),
            'confidence': sel['confidence'],
            'success_rate': sel['confidence'] * 0.9,
            'last_updated': datetime.now().isoformat(),
            'context': json.dumps({'platform': 'google', 'action': sel['action']})
        })
    
    return selectors

def create_github_selectors():
    """Create comprehensive GitHub selectors"""
    selectors = []
    
    github_selectors = [
        # Repository actions
        {'selector': '[data-testid="new-repo-button"]', 'type': 'css', 'action': 'click', 'confidence': 0.95},
        {'selector': '.btn-primary', 'type': 'css', 'action': 'click', 'confidence': 0.90},
        {'selector': '.btn', 'type': 'css', 'action': 'click', 'confidence': 0.85},
        
        # Search
        {'selector': 'input[name="q"]', 'type': 'css', 'action': 'type', 'confidence': 0.95},
        {'selector': '[placeholder="Search GitHub"]', 'type': 'css', 'action': 'type', 'confidence': 0.90},
        
        # Navigation
        {'selector': '.Header-link', 'type': 'css', 'action': 'click', 'confidence': 0.85},
        {'selector': '.js-navigation-open', 'type': 'css', 'action': 'click', 'confidence': 0.90},
        
        # Repository management
        {'selector': '#repository_name', 'type': 'css', 'action': 'type', 'confidence': 0.95},
        {'selector': '.form-control', 'type': 'css', 'action': 'type', 'confidence': 0.85},
        
        # File operations
        {'selector': '.js-navigation-open', 'type': 'css', 'action': 'click', 'confidence': 0.80},
        {'selector': '.octicon-plus', 'type': 'css', 'action': 'click', 'confidence': 0.85}
    ]
    
    for i, sel in enumerate(github_selectors):
        selectors.append({
            'id': i + 1,
            'platform': 'github',
            'category': 'developer',
            'module': 'repositories',
            'selector_type': sel['type'],
            'action_type': sel['action'],
            'complexity': 'complex',
            'selector': sel['selector'],
            'fallback_selectors': json.dumps([]),
            'confidence': sel['confidence'],
            'success_rate': sel['confidence'] * 0.9,
            'last_updated': datetime.now().isoformat(),
            'context': json.dumps({'platform': 'github', 'action': sel['action']})
        })
    
    return selectors

def create_aws_selectors():
    """Create comprehensive AWS selectors"""
    selectors = []
    
    aws_selectors = [
        # AWS Console navigation
        {'selector': '.awsui-button', 'type': 'css', 'action': 'click', 'confidence': 0.95},
        {'selector': '[data-testid="awsui-button"]', 'type': 'css', 'action': 'click', 'confidence': 0.90},
        {'selector': '.btn-primary', 'type': 'css', 'action': 'click', 'confidence': 0.85},
        
        # Search and navigation
        {'selector': 'input[type="text"]', 'type': 'css', 'action': 'type', 'confidence': 0.90},
        {'selector': '.awsui-input', 'type': 'css', 'action': 'type', 'confidence': 0.85},
        
        # Service navigation
        {'selector': '.service-link', 'type': 'css', 'action': 'click', 'confidence': 0.80},
        {'selector': '[role="menuitem"]', 'type': 'css', 'action': 'click', 'confidence': 0.85},
        
        # Form controls
        {'selector': 'select', 'type': 'css', 'action': 'select', 'confidence': 0.90},
        {'selector': '.awsui-select', 'type': 'css', 'action': 'select', 'confidence': 0.85},
        
        # Instance management
        {'selector': '.instance-checkbox', 'type': 'css', 'action': 'click', 'confidence': 0.80},
        {'selector': '[data-testid="checkbox"]', 'type': 'css', 'action': 'click', 'confidence': 0.75}
    ]
    
    for i, sel in enumerate(aws_selectors):
        selectors.append({
            'id': i + 1,
            'platform': 'aws',
            'category': 'developer',
            'module': 'console',
            'selector_type': sel['type'],
            'action_type': sel['action'],
            'complexity': 'ultra_complex',
            'selector': sel['selector'],
            'fallback_selectors': json.dumps([]),
            'confidence': sel['confidence'],
            'success_rate': sel['confidence'] * 0.9,
            'last_updated': datetime.now().isoformat(),
            'context': json.dumps({'platform': 'aws', 'action': sel['action']})
        })
    
    return selectors

def create_database(platform_name, category, selectors):
    """Create database for platform selectors"""
    db_name = f"ultra_selectors_{category}_{platform_name}.db"
    
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    # Create table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS selectors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            platform TEXT NOT NULL,
            category TEXT NOT NULL,
            module TEXT NOT NULL,
            selector_type TEXT NOT NULL,
            action_type TEXT NOT NULL,
            complexity TEXT NOT NULL,
            selector TEXT NOT NULL,
            fallback_selectors TEXT NOT NULL,
            confidence REAL NOT NULL,
            success_rate REAL NOT NULL,
            last_updated TEXT NOT NULL,
            context TEXT NOT NULL,
            usage_count INTEGER DEFAULT 0,
            success_count INTEGER DEFAULT 0,
            failure_count INTEGER DEFAULT 0,
            created_date TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Insert selectors
    for selector_data in selectors:
        cursor.execute('''
            INSERT INTO selectors (
                platform, category, module, selector_type, action_type,
                complexity, selector, fallback_selectors, confidence,
                success_rate, last_updated, context
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            selector_data['platform'],
            selector_data['category'],
            selector_data['module'],
            selector_data['selector_type'],
            selector_data['action_type'],
            selector_data['complexity'],
            selector_data['selector'],
            selector_data['fallback_selectors'],
            selector_data['confidence'],
            selector_data['success_rate'],
            selector_data['last_updated'],
            selector_data['context']
        ))
    
    # Create indexes
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_platform ON selectors(platform)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_action ON selectors(action_type)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_complexity ON selectors(complexity)')
    
    conn.commit()
    conn.close()
    
    return db_name, len(selectors)

def update_master_index(new_databases):
    """Update master index with new databases"""
    conn = sqlite3.connect("ultra_selector_master_index.db")
    cursor = conn.cursor()
    
    for db_info in new_databases:
        # Check if already exists
        cursor.execute(
            "SELECT COUNT(*) FROM platform_databases WHERE database_file = ?",
            (db_info['database'],)
        )
        
        if cursor.fetchone()[0] == 0:
            cursor.execute('''
                INSERT INTO platform_databases (
                    database_file, platform_name, category, selector_count, complexity
                ) VALUES (?, ?, ?, ?, ?)
            ''', (
                db_info['database'],
                db_info['platform'],
                db_info['category'],
                db_info['selector_count'],
                db_info['complexity']
            ))
    
    conn.commit()
    conn.close()

def main():
    """Create missing platform selectors"""
    print("🚀 CREATING MISSING PLATFORM SELECTORS")
    print("=" * 60)
    
    new_databases = []
    
    # Create Google selectors
    print("📂 Creating Google selectors...")
    google_selectors = create_google_selectors()
    db_name, count = create_database('google', 'search', google_selectors)
    new_databases.append({
        'database': db_name,
        'platform': 'Google',
        'category': 'search',
        'selector_count': count,
        'complexity': 'simple'
    })
    print(f"   ✅ Created {db_name} with {count} selectors")
    
    # Create GitHub selectors
    print("📂 Creating GitHub selectors...")
    github_selectors = create_github_selectors()
    db_name, count = create_database('github', 'developer', github_selectors)
    new_databases.append({
        'database': db_name,
        'platform': 'GitHub',
        'category': 'developer',
        'selector_count': count,
        'complexity': 'complex'
    })
    print(f"   ✅ Created {db_name} with {count} selectors")
    
    # Create AWS selectors
    print("📂 Creating AWS selectors...")
    aws_selectors = create_aws_selectors()
    db_name, count = create_database('aws_console', 'cloud', aws_selectors)
    new_databases.append({
        'database': db_name,
        'platform': 'AWS Console',
        'category': 'cloud',
        'selector_count': count,
        'complexity': 'ultra_complex'
    })
    print(f"   ✅ Created {db_name} with {count} selectors")
    
    # Update master index
    print("📋 Updating master index...")
    update_master_index(new_databases)
    print("   ✅ Master index updated")
    
    total_selectors = sum(db['selector_count'] for db in new_databases)
    print(f"\n🎉 MISSING PLATFORM SELECTORS CREATED!")
    print(f"📊 New Databases: {len(new_databases)}")
    print(f"📊 Total New Selectors: {total_selectors}")
    print(f"🎯 Platform Coverage Enhanced!")

if __name__ == "__main__":
    main()