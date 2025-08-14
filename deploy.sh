#!/bin/bash

# SUPER-OMEGA Production Deployment Script
# ========================================
# Next-Generation AI-First Automation Platform
# Superior to UiPath, Automation Anywhere, Manus AI

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SUPER_OMEGA_VERSION="2.0.0"
PYTHON_VERSION="3.11"
NODE_VERSION="18"
INSTALL_DIR="/opt/super-omega"
DATA_DIR="/var/lib/super-omega"
LOG_DIR="/var/log/super-omega"
CONFIG_DIR="/etc/super-omega"

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${PURPLE}================================${NC}"
    echo -e "${PURPLE}$1${NC}"
    echo -e "${PURPLE}================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        print_error "This script should not be run as root for security reasons"
        print_status "Please run as a regular user with sudo privileges"
        exit 1
    fi
}

# Check system requirements
check_system_requirements() {
    print_header "CHECKING SYSTEM REQUIREMENTS"
    
    # Check OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        print_success "Linux system detected"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        print_success "macOS system detected"
    else
        print_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
    
    # Check memory
    TOTAL_MEM=$(free -m 2>/dev/null | awk 'NR==2{printf "%.0f", $2}' || sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.0f", $1/1024/1024}')
    if [[ $TOTAL_MEM -lt 8192 ]]; then
        print_warning "Minimum 8GB RAM recommended, detected ${TOTAL_MEM}MB"
    else
        print_success "Memory check passed: ${TOTAL_MEM}MB"
    fi
    
    # Check disk space
    AVAILABLE_SPACE=$(df / | awk 'NR==2 {print $4}')
    if [[ $AVAILABLE_SPACE -lt 10485760 ]]; then  # 10GB in KB
        print_error "Minimum 10GB free disk space required"
        exit 1
    else
        print_success "Disk space check passed"
    fi
    
    # Check internet connectivity
    if ping -c 1 google.com &> /dev/null; then
        print_success "Internet connectivity verified"
    else
        print_error "Internet connection required for installation"
        exit 1
    fi
}

# Install system dependencies
install_system_dependencies() {
    print_header "INSTALLING SYSTEM DEPENDENCIES"
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Detect Linux distribution
        if [ -f /etc/debian_version ]; then
            print_status "Installing dependencies for Debian/Ubuntu..."
            sudo apt-get update
            sudo apt-get install -y \
                python3.11 python3.11-dev python3.11-venv \
                python3-pip nodejs npm \
                git curl wget \
                build-essential \
                libssl-dev libffi-dev \
                postgresql postgresql-contrib \
                redis-server \
                nginx \
                supervisor \
                htop tmux \
                libnss3-dev libatk-bridge2.0-dev libdrm2 \
                libxcomposite1 libxdamage1 libxrandr2 \
                libgbm1 libxss1 libasound2
        elif [ -f /etc/redhat-release ]; then
            print_status "Installing dependencies for RHEL/CentOS..."
            sudo yum update -y
            sudo yum install -y \
                python311 python311-devel python311-pip \
                nodejs npm \
                git curl wget \
                gcc gcc-c++ make \
                openssl-devel libffi-devel \
                postgresql postgresql-server \
                redis \
                nginx \
                supervisor \
                htop tmux
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        print_status "Installing dependencies for macOS..."
        if ! command -v brew &> /dev/null; then
            print_status "Installing Homebrew..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        
        brew update
        brew install \
            python@3.11 \
            node \
            postgresql \
            redis \
            nginx \
            supervisor \
            htop \
            tmux
    fi
    
    print_success "System dependencies installed"
}

# Install Python dependencies
install_python_dependencies() {
    print_header "SETTING UP PYTHON ENVIRONMENT"
    
    # Create virtual environment
    print_status "Creating Python virtual environment..."
    python3.11 -m venv ~/.super-omega-venv
    source ~/.super-omega-venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    # Install SUPER-OMEGA dependencies
    print_status "Installing SUPER-OMEGA Python dependencies..."
    pip install -r requirements.txt
    
    # Install Playwright browsers
    print_status "Installing Playwright browsers..."
    playwright install chromium firefox webkit
    playwright install-deps
    
    print_success "Python environment configured"
}

# Setup directories
setup_directories() {
    print_header "CREATING DIRECTORY STRUCTURE"
    
    # Create directories
    sudo mkdir -p $INSTALL_DIR $DATA_DIR $LOG_DIR $CONFIG_DIR
    sudo mkdir -p $DATA_DIR/{evidence,skills,models,audit,keys}
    sudo mkdir -p $LOG_DIR/{app,nginx,supervisor}
    
    # Set permissions
    sudo chown -R $USER:$USER $INSTALL_DIR $DATA_DIR $LOG_DIR
    sudo chmod 755 $INSTALL_DIR $DATA_DIR $LOG_DIR
    sudo chmod 700 $DATA_DIR/keys  # Restrict key directory
    
    print_success "Directory structure created"
}

# Setup database
setup_database() {
    print_header "CONFIGURING DATABASE"
    
    # PostgreSQL setup
    print_status "Setting up PostgreSQL..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo systemctl start postgresql
        sudo systemctl enable postgresql
        
        # Create database and user
        sudo -u postgres createdb super_omega 2>/dev/null || true
        sudo -u postgres psql -c "CREATE USER super_omega WITH PASSWORD 'super_omega_secure_pass';" 2>/dev/null || true
        sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE super_omega TO super_omega;" 2>/dev/null || true
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        brew services start postgresql
        createdb super_omega 2>/dev/null || true
        psql super_omega -c "CREATE USER super_omega WITH PASSWORD 'super_omega_secure_pass';" 2>/dev/null || true
    fi
    
    # Redis setup
    print_status "Setting up Redis..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo systemctl start redis
        sudo systemctl enable redis
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        brew services start redis
    fi
    
    print_success "Database configured"
}

# Setup configuration files
setup_configuration() {
    print_header "CREATING CONFIGURATION FILES"
    
    # Main configuration
    cat > $CONFIG_DIR/super-omega.yaml << EOF
# SUPER-OMEGA Production Configuration
version: "$SUPER_OMEGA_VERSION"

# Server settings
server:
  host: "0.0.0.0"
  port: 8080
  workers: 4
  
# Database settings
database:
  url: "postgresql://super_omega:super_omega_secure_pass@localhost/super_omega"
  
# Redis settings
redis:
  url: "redis://localhost:6379/0"
  
# Security settings
security:
  secret_key: "$(openssl rand -base64 32)"
  encryption_key: "$(openssl rand -base64 32)"
  session_timeout: 28800  # 8 hours
  
# AI settings
ai:
  openai_api_key: "${OPENAI_API_KEY:-}"
  anthropic_api_key: "${ANTHROPIC_API_KEY:-}"
  
# Performance settings
performance:
  max_parallel_steps: 10
  step_timeout_ms: 15000
  plan_timeout_ms: 120000
  
# Evidence settings
evidence:
  base_dir: "$DATA_DIR/evidence"
  capture_screenshots: true
  capture_video: true
  retention_days: 90
  
# Skill mining settings
skill_mining:
  enabled: true
  confidence_threshold: 0.9
  min_validation_runs: 3
  
# Logging
logging:
  level: "INFO"
  file: "$LOG_DIR/app/super-omega.log"
  max_size: "100MB"
  backup_count: 10
EOF

    # Nginx configuration
    sudo tee $CONFIG_DIR/nginx.conf > /dev/null << EOF
server {
    listen 80;
    server_name _;
    
    # Redirect HTTP to HTTPS
    return 301 https://\$server_name\$request_uri;
}

server {
    listen 443 ssl http2;
    server_name _;
    
    # SSL configuration (update with your certificates)
    ssl_certificate /etc/ssl/certs/super-omega.crt;
    ssl_certificate_key /etc/ssl/private/super-omega.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload";
    
    # Rate limiting
    limit_req_zone \$binary_remote_addr zone=api:10m rate=10r/s;
    
    location / {
        limit_req zone=api burst=20 nodelay;
        
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    # Static files
    location /static/ {
        alias $DATA_DIR/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
EOF

    # Supervisor configuration
    sudo tee $CONFIG_DIR/supervisor.conf > /dev/null << EOF
[program:super-omega]
command=$HOME/.super-omega-venv/bin/python main.py
directory=$INSTALL_DIR
user=$USER
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=$LOG_DIR/supervisor/super-omega.log
stdout_logfile_maxbytes=50MB
stdout_logfile_backups=10
environment=PATH="$HOME/.super-omega-venv/bin",PYTHONPATH="$INSTALL_DIR"
EOF

    # Systemd service
    sudo tee /etc/systemd/system/super-omega.service > /dev/null << EOF
[Unit]
Description=SUPER-OMEGA Automation Platform
After=network.target postgresql.service redis.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$INSTALL_DIR
Environment=PATH=$HOME/.super-omega-venv/bin
Environment=PYTHONPATH=$INSTALL_DIR
ExecStart=$HOME/.super-omega-venv/bin/python main.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

    print_success "Configuration files created"
}

# Setup SSL certificates
setup_ssl() {
    print_header "SETTING UP SSL CERTIFICATES"
    
    if [ ! -f /etc/ssl/certs/super-omega.crt ]; then
        print_status "Generating self-signed SSL certificate..."
        sudo openssl req -x509 -nodes -days 365 -newkey rsa:4096 \
            -keyout /etc/ssl/private/super-omega.key \
            -out /etc/ssl/certs/super-omega.crt \
            -subj "/C=US/ST=State/L=City/O=Organization/CN=super-omega"
        
        sudo chmod 600 /etc/ssl/private/super-omega.key
        sudo chmod 644 /etc/ssl/certs/super-omega.crt
        
        print_warning "Self-signed certificate generated. Replace with proper certificate for production."
    else
        print_success "SSL certificate already exists"
    fi
}

# Deploy application
deploy_application() {
    print_header "DEPLOYING APPLICATION"
    
    # Copy application files
    print_status "Copying application files..."
    sudo cp -r . $INSTALL_DIR/
    sudo chown -R $USER:$USER $INSTALL_DIR
    
    # Install application
    cd $INSTALL_DIR
    source ~/.super-omega-venv/bin/activate
    pip install -e .
    
    print_success "Application deployed"
}

# Setup monitoring
setup_monitoring() {
    print_header "SETTING UP MONITORING"
    
    # Create monitoring script
    cat > $INSTALL_DIR/monitor.py << 'EOF'
#!/usr/bin/env python3
"""
SUPER-OMEGA System Monitor
"""
import psutil
import requests
import time
import json
from datetime import datetime

def check_system_health():
    """Check system health metrics."""
    health = {
        'timestamp': datetime.utcnow().isoformat(),
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_percent': psutil.disk_usage('/').percent,
        'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
    }
    
    # Check SUPER-OMEGA API
    try:
        response = requests.get('http://localhost:8080/', timeout=5)
        health['api_status'] = 'healthy' if response.status_code == 200 else 'unhealthy'
    except:
        health['api_status'] = 'unreachable'
    
    return health

if __name__ == '__main__':
    health = check_system_health()
    print(json.dumps(health, indent=2))
EOF

    chmod +x $INSTALL_DIR/monitor.py
    
    # Create monitoring cron job
    (crontab -l 2>/dev/null; echo "*/5 * * * * $INSTALL_DIR/monitor.py >> $LOG_DIR/monitoring.log") | crontab -
    
    print_success "Monitoring configured"
}

# Setup firewall
setup_firewall() {
    print_header "CONFIGURING FIREWALL"
    
    if command -v ufw &> /dev/null; then
        print_status "Configuring UFW firewall..."
        sudo ufw --force enable
        sudo ufw default deny incoming
        sudo ufw default allow outgoing
        sudo ufw allow ssh
        sudo ufw allow 80/tcp
        sudo ufw allow 443/tcp
        print_success "UFW firewall configured"
    elif command -v firewall-cmd &> /dev/null; then
        print_status "Configuring firewalld..."
        sudo systemctl start firewalld
        sudo systemctl enable firewalld
        sudo firewall-cmd --permanent --add-service=http
        sudo firewall-cmd --permanent --add-service=https
        sudo firewall-cmd --permanent --add-service=ssh
        sudo firewall-cmd --reload
        print_success "Firewalld configured"
    else
        print_warning "No firewall detected. Please configure manually."
    fi
}

# Start services
start_services() {
    print_header "STARTING SERVICES"
    
    # Enable and start systemd service
    sudo systemctl daemon-reload
    sudo systemctl enable super-omega
    sudo systemctl start super-omega
    
    # Start nginx
    sudo systemctl enable nginx
    sudo systemctl start nginx
    
    # Wait for services to start
    sleep 5
    
    # Check service status
    if sudo systemctl is-active --quiet super-omega; then
        print_success "SUPER-OMEGA service started"
    else
        print_error "Failed to start SUPER-OMEGA service"
        sudo systemctl status super-omega
        exit 1
    fi
    
    if sudo systemctl is-active --quiet nginx; then
        print_success "Nginx service started"
    else
        print_error "Failed to start Nginx service"
        sudo systemctl status nginx
        exit 1
    fi
}

# Verify installation
verify_installation() {
    print_header "VERIFYING INSTALLATION"
    
    # Check API endpoint
    print_status "Testing API endpoint..."
    sleep 10  # Wait for full startup
    
    if curl -f -s http://localhost:8080/ > /dev/null; then
        print_success "API endpoint responding"
    else
        print_error "API endpoint not responding"
        exit 1
    fi
    
    # Check live console
    if curl -f -s http://localhost:8080/console > /dev/null; then
        print_success "Live console accessible"
    else
        print_warning "Live console may not be accessible"
    fi
    
    # Check metrics endpoint
    if curl -f -s http://localhost:8080/api/metrics > /dev/null; then
        print_success "Metrics endpoint responding"
    else
        print_warning "Metrics endpoint may not be accessible"
    fi
}

# Print deployment summary
print_deployment_summary() {
    print_header "DEPLOYMENT COMPLETE"
    
    echo -e "${GREEN}🚀 SUPER-OMEGA Successfully Deployed!${NC}"
    echo ""
    echo -e "${CYAN}System Information:${NC}"
    echo "  Version: $SUPER_OMEGA_VERSION"
    echo "  Install Directory: $INSTALL_DIR"
    echo "  Data Directory: $DATA_DIR"
    echo "  Log Directory: $LOG_DIR"
    echo ""
    echo -e "${CYAN}Access URLs:${NC}"
    echo "  Main API: http://localhost:8080/"
    echo "  Live Console: http://localhost:8080/console"
    echo "  API Documentation: http://localhost:8080/api/docs"
    echo ""
    echo -e "${CYAN}Management Commands:${NC}"
    echo "  Start: sudo systemctl start super-omega"
    echo "  Stop: sudo systemctl stop super-omega"
    echo "  Restart: sudo systemctl restart super-omega"
    echo "  Status: sudo systemctl status super-omega"
    echo "  Logs: sudo journalctl -u super-omega -f"
    echo ""
    echo -e "${CYAN}Configuration:${NC}"
    echo "  Main Config: $CONFIG_DIR/super-omega.yaml"
    echo "  Nginx Config: $CONFIG_DIR/nginx.conf"
    echo ""
    echo -e "${YELLOW}Next Steps:${NC}"
    echo "  1. Set your API keys in $CONFIG_DIR/super-omega.yaml"
    echo "  2. Replace self-signed SSL certificate with proper certificate"
    echo "  3. Configure your firewall rules"
    echo "  4. Set up backup procedures for $DATA_DIR"
    echo "  5. Monitor logs and metrics"
    echo ""
    echo -e "${GREEN}SUPER-OMEGA is ready to outperform all RPA platforms!${NC}"
}

# Main deployment function
main() {
    print_header "SUPER-OMEGA PRODUCTION DEPLOYMENT"
    echo -e "${PURPLE}Next-Generation AI-First Automation Platform${NC}"
    echo -e "${PURPLE}Superior to UiPath, Automation Anywhere, Manus AI${NC}"
    echo ""
    
    check_root
    check_system_requirements
    install_system_dependencies
    install_python_dependencies
    setup_directories
    setup_database
    setup_configuration
    setup_ssl
    deploy_application
    setup_monitoring
    setup_firewall
    start_services
    verify_installation
    print_deployment_summary
}

# Run main function
main "$@"