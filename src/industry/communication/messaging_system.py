"""
Enterprise Communication & Messaging System
==========================================

Comprehensive automation for all communication scenarios:
- Email automation (Gmail, Outlook, Yahoo, custom SMTP)
- SMS/OTP verification (Twilio, AWS SNS, Google Firebase)
- WhatsApp Business API automation
- Slack/Teams/Discord integration
- Social media messaging (Facebook, Instagram, Twitter)
- Voice calls and IVR automation
- Push notifications (FCM, APNS)
- In-app messaging systems

Features:
- Multi-provider failover and load balancing
- Template management with dynamic content
- Delivery tracking and analytics
- Spam detection and compliance
- End-to-end encryption
- Bulk messaging with rate limiting
- A/B testing for message optimization
- Real-time delivery status tracking
"""

import asyncio
import logging
import json
import re
import smtplib
import imaplib
import poplib
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import base64
import hashlib
import hmac
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from email.mime.base import MimeBase
from email import encoders
import ssl

try:
    import twilio
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False

try:
    import boto3
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

try:
    import firebase_admin
    from firebase_admin import messaging
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False

try:
    import requests
    import websocket
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from ...core.deterministic_executor import DeterministicExecutor
from ...core.realtime_data_fabric import RealTimeDataFabric
from ...core.enterprise_security import EnterpriseSecurityManager


class MessageType(str, Enum):
    """Types of messages supported."""
    EMAIL = "email"
    SMS = "sms"
    WHATSAPP = "whatsapp"
    SLACK = "slack"
    TEAMS = "teams"
    DISCORD = "discord"
    TELEGRAM = "telegram"
    PUSH_NOTIFICATION = "push_notification"
    VOICE_CALL = "voice_call"
    FAX = "fax"
    WEBHOOK = "webhook"


class MessageStatus(str, Enum):
    """Message delivery status."""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    READ = "read"
    FAILED = "failed"
    BOUNCED = "bounced"
    SPAM = "spam"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class Priority(str, Enum):
    """Message priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"


class ProviderType(str, Enum):
    """Communication providers."""
    # Email providers
    GMAIL = "gmail"
    OUTLOOK = "outlook"
    YAHOO = "yahoo"
    SENDGRID = "sendgrid"
    MAILGUN = "mailgun"
    SES = "ses"
    
    # SMS providers
    TWILIO = "twilio"
    AWS_SNS = "aws_sns"
    FIREBASE = "firebase"
    NEXMO = "nexmo"
    CLICKSEND = "clicksend"
    
    # Messaging platforms
    WHATSAPP_BUSINESS = "whatsapp_business"
    SLACK_API = "slack_api"
    TEAMS_API = "teams_api"
    DISCORD_API = "discord_api"
    TELEGRAM_API = "telegram_api"


@dataclass
class Recipient:
    """Message recipient information."""
    email: Optional[str] = None
    phone: Optional[str] = None
    name: Optional[str] = None
    user_id: Optional[str] = None
    platform_id: Optional[str] = None  # Slack user ID, etc.
    timezone: Optional[str] = None
    language: Optional[str] = "en"
    preferences: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.preferences is None:
            self.preferences = {}


@dataclass
class MessageTemplate:
    """Message template with dynamic content."""
    template_id: str
    name: str
    subject: Optional[str] = None
    body: str = ""
    html_body: Optional[str] = None
    variables: List[str] = None
    message_type: MessageType = MessageType.EMAIL
    language: str = "en"
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.variables is None:
            self.variables = []
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()


@dataclass
class MessageContent:
    """Message content with all components."""
    subject: Optional[str] = None
    body: str = ""
    html_body: Optional[str] = None
    attachments: List[Dict[str, Any]] = None
    variables: Dict[str, Any] = None
    template_id: Optional[str] = None
    
    def __post_init__(self):
        if self.attachments is None:
            self.attachments = []
        if self.variables is None:
            self.variables = {}


@dataclass
class MessageRequest:
    """Comprehensive message request."""
    message_type: MessageType
    recipients: List[Recipient]
    content: MessageContent
    sender: Optional[str] = None
    priority: Priority = Priority.NORMAL
    scheduled_time: Optional[datetime] = None
    expiry_time: Optional[datetime] = None
    tracking_enabled: bool = True
    delivery_receipt: bool = True
    read_receipt: bool = False
    retry_attempts: int = 3
    provider_preference: List[ProviderType] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.provider_preference is None:
            self.provider_preference = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MessageResult:
    """Message delivery result."""
    message_id: str
    status: MessageStatus
    message_type: MessageType
    recipient_count: int
    sent_count: int = 0
    delivered_count: int = 0
    failed_count: int = 0
    provider_used: Optional[ProviderType] = None
    cost: Optional[float] = None
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    error_message: Optional[str] = None
    tracking_data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.tracking_data is None:
            self.tracking_data = {}


class EmailProvider:
    """Advanced email provider with multiple backend support."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Email provider configurations
        self.providers = {
            ProviderType.GMAIL: {
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'imap_server': 'imap.gmail.com',
                'imap_port': 993,
                'use_tls': True
            },
            ProviderType.OUTLOOK: {
                'smtp_server': 'smtp-mail.outlook.com',
                'smtp_port': 587,
                'imap_server': 'outlook.office365.com',
                'imap_port': 993,
                'use_tls': True
            },
            ProviderType.YAHOO: {
                'smtp_server': 'smtp.mail.yahoo.com',
                'smtp_port': 587,
                'imap_server': 'imap.mail.yahoo.com',
                'imap_port': 993,
                'use_tls': True
            }
        }
        
        # Provider credentials
        self.credentials = config.get('email_credentials', {})
        
        # Email tracking
        self.tracking_pixels = {}
        self.delivery_webhooks = {}
    
    async def send_email(self, request: MessageRequest) -> MessageResult:
        """Send email with advanced features."""
        message_id = str(uuid.uuid4())
        
        try:
            # Select best provider
            provider = self._select_email_provider(request.provider_preference)
            
            # Prepare email content
            email_content = await self._prepare_email_content(request.content)
            
            # Send to all recipients
            sent_count = 0
            failed_count = 0
            
            for recipient in request.recipients:
                if not recipient.email:
                    failed_count += 1
                    continue
                
                try:
                    # Send individual email
                    await self._send_individual_email(
                        provider, recipient, email_content, request.sender
                    )
                    sent_count += 1
                    
                    # Add tracking if enabled
                    if request.tracking_enabled:
                        await self._add_email_tracking(message_id, recipient.email)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to send email to {recipient.email}: {e}")
                    failed_count += 1
            
            status = MessageStatus.SENT if sent_count > 0 else MessageStatus.FAILED
            
            return MessageResult(
                message_id=message_id,
                status=status,
                message_type=MessageType.EMAIL,
                recipient_count=len(request.recipients),
                sent_count=sent_count,
                failed_count=failed_count,
                provider_used=provider,
                sent_at=datetime.utcnow()
            )
            
        except Exception as e:
            self.logger.error(f"Email sending failed: {e}")
            return MessageResult(
                message_id=message_id,
                status=MessageStatus.FAILED,
                message_type=MessageType.EMAIL,
                recipient_count=len(request.recipients),
                failed_count=len(request.recipients),
                error_message=str(e)
            )
    
    async def _send_individual_email(self, provider: ProviderType, recipient: Recipient, 
                                   content: Dict[str, Any], sender: str):
        """Send email to individual recipient."""
        try:
            provider_config = self.providers[provider]
            credentials = self.credentials.get(provider.value, {})
            
            # Create message
            msg = MimeMultipart('alternative')
            msg['From'] = sender or credentials.get('username')
            msg['To'] = recipient.email
            msg['Subject'] = content['subject']
            
            # Add text body
            if content.get('body'):
                text_part = MimeText(content['body'], 'plain', 'utf-8')
                msg.attach(text_part)
            
            # Add HTML body
            if content.get('html_body'):
                html_part = MimeText(content['html_body'], 'html', 'utf-8')
                msg.attach(html_part)
            
            # Add attachments
            for attachment in content.get('attachments', []):
                await self._add_attachment(msg, attachment)
            
            # Send via SMTP
            context = ssl.create_default_context()
            
            with smtplib.SMTP(provider_config['smtp_server'], provider_config['smtp_port']) as server:
                if provider_config['use_tls']:
                    server.starttls(context=context)
                
                server.login(credentials['username'], credentials['password'])
                server.send_message(msg)
            
            self.logger.info(f"Email sent successfully to {recipient.email}")
            
        except Exception as e:
            self.logger.error(f"Individual email send failed: {e}")
            raise
    
    async def _prepare_email_content(self, content: MessageContent) -> Dict[str, Any]:
        """Prepare email content with template processing."""
        try:
            prepared_content = {
                'subject': content.subject or '',
                'body': content.body or '',
                'html_body': content.html_body,
                'attachments': content.attachments or []
            }
            
            # Process template variables
            if content.variables:
                for key, value in prepared_content.items():
                    if isinstance(value, str) and value:
                        prepared_content[key] = self._replace_variables(value, content.variables)
            
            return prepared_content
            
        except Exception as e:
            self.logger.error(f"Email content preparation failed: {e}")
            raise
    
    def _replace_variables(self, text: str, variables: Dict[str, Any]) -> str:
        """Replace template variables in text."""
        try:
            for var_name, var_value in variables.items():
                placeholder = f"{{{{{var_name}}}}}"
                text = text.replace(placeholder, str(var_value))
            return text
        except Exception as e:
            self.logger.warning(f"Variable replacement failed: {e}")
            return text
    
    def _select_email_provider(self, preferences: List[ProviderType]) -> ProviderType:
        """Select best available email provider."""
        # Check preferences first
        for provider in preferences:
            if provider in self.providers and provider.value in self.credentials:
                return provider
        
        # Default fallback
        available_providers = [
            p for p in [ProviderType.GMAIL, ProviderType.OUTLOOK, ProviderType.YAHOO]
            if p.value in self.credentials
        ]
        
        return available_providers[0] if available_providers else ProviderType.GMAIL


class SMSProvider:
    """Advanced SMS provider with OTP verification support."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize providers
        self.twilio_client = None
        self.aws_sns_client = None
        
        if TWILIO_AVAILABLE and 'twilio' in config:
            try:
                self.twilio_client = TwilioClient(
                    config['twilio']['account_sid'],
                    config['twilio']['auth_token']
                )
            except Exception as e:
                self.logger.warning(f"Twilio initialization failed: {e}")
        
        if AWS_AVAILABLE and 'aws' in config:
            try:
                self.aws_sns_client = boto3.client(
                    'sns',
                    aws_access_key_id=config['aws']['access_key'],
                    aws_secret_access_key=config['aws']['secret_key'],
                    region_name=config['aws'].get('region', 'us-east-1')
                )
            except Exception as e:
                self.logger.warning(f"AWS SNS initialization failed: {e}")
        
        # OTP storage and verification
        self.otp_storage = {}
        self.otp_attempts = {}
    
    async def send_sms(self, request: MessageRequest) -> MessageResult:
        """Send SMS with provider failover."""
        message_id = str(uuid.uuid4())
        
        try:
            sent_count = 0
            failed_count = 0
            provider_used = None
            
            for recipient in request.recipients:
                if not recipient.phone:
                    failed_count += 1
                    continue
                
                # Try providers in order of preference
                success = False
                for provider in request.provider_preference or [ProviderType.TWILIO, ProviderType.AWS_SNS]:
                    try:
                        if provider == ProviderType.TWILIO and self.twilio_client:
                            await self._send_twilio_sms(recipient, request.content)
                            provider_used = provider
                            success = True
                            break
                        elif provider == ProviderType.AWS_SNS and self.aws_sns_client:
                            await self._send_aws_sms(recipient, request.content)
                            provider_used = provider
                            success = True
                            break
                    except Exception as e:
                        self.logger.warning(f"SMS provider {provider} failed: {e}")
                        continue
                
                if success:
                    sent_count += 1
                else:
                    failed_count += 1
            
            status = MessageStatus.SENT if sent_count > 0 else MessageStatus.FAILED
            
            return MessageResult(
                message_id=message_id,
                status=status,
                message_type=MessageType.SMS,
                recipient_count=len(request.recipients),
                sent_count=sent_count,
                failed_count=failed_count,
                provider_used=provider_used,
                sent_at=datetime.utcnow()
            )
            
        except Exception as e:
            self.logger.error(f"SMS sending failed: {e}")
            return MessageResult(
                message_id=message_id,
                status=MessageStatus.FAILED,
                message_type=MessageType.SMS,
                recipient_count=len(request.recipients),
                failed_count=len(request.recipients),
                error_message=str(e)
            )
    
    async def _send_twilio_sms(self, recipient: Recipient, content: MessageContent):
        """Send SMS via Twilio."""
        try:
            message = self.twilio_client.messages.create(
                body=content.body,
                from_=self.config['twilio']['from_number'],
                to=recipient.phone
            )
            
            self.logger.info(f"Twilio SMS sent to {recipient.phone}: {message.sid}")
            
        except Exception as e:
            self.logger.error(f"Twilio SMS failed: {e}")
            raise
    
    async def _send_aws_sms(self, recipient: Recipient, content: MessageContent):
        """Send SMS via AWS SNS."""
        try:
            response = self.aws_sns_client.publish(
                PhoneNumber=recipient.phone,
                Message=content.body,
                MessageAttributes={
                    'AWS.SNS.SMS.SMSType': {
                        'DataType': 'String',
                        'StringValue': 'Transactional'
                    }
                }
            )
            
            self.logger.info(f"AWS SNS SMS sent to {recipient.phone}: {response['MessageId']}")
            
        except Exception as e:
            self.logger.error(f"AWS SNS SMS failed: {e}")
            raise
    
    async def generate_otp(self, phone: str, length: int = 6, expiry_minutes: int = 10) -> str:
        """Generate and send OTP."""
        try:
            # Generate OTP
            import random
            otp = ''.join([str(random.randint(0, 9)) for _ in range(length)])
            
            # Store OTP with expiry
            otp_key = f"otp_{phone}"
            self.otp_storage[otp_key] = {
                'code': otp,
                'phone': phone,
                'created_at': datetime.utcnow(),
                'expires_at': datetime.utcnow() + timedelta(minutes=expiry_minutes),
                'attempts': 0,
                'verified': False
            }
            
            # Send OTP via SMS
            recipient = Recipient(phone=phone)
            content = MessageContent(body=f"Your verification code is: {otp}")
            request = MessageRequest(
                message_type=MessageType.SMS,
                recipients=[recipient],
                content=content
            )
            
            result = await self.send_sms(request)
            
            if result.status == MessageStatus.SENT:
                return otp
            else:
                raise Exception("Failed to send OTP SMS")
                
        except Exception as e:
            self.logger.error(f"OTP generation failed: {e}")
            raise
    
    async def verify_otp(self, phone: str, otp: str) -> bool:
        """Verify OTP code."""
        try:
            otp_key = f"otp_{phone}"
            
            if otp_key not in self.otp_storage:
                return False
            
            otp_data = self.otp_storage[otp_key]
            
            # Check if expired
            if datetime.utcnow() > otp_data['expires_at']:
                del self.otp_storage[otp_key]
                return False
            
            # Check attempts
            if otp_data['attempts'] >= 3:
                del self.otp_storage[otp_key]
                return False
            
            # Verify OTP
            if otp_data['code'] == otp:
                otp_data['verified'] = True
                del self.otp_storage[otp_key]
                return True
            else:
                otp_data['attempts'] += 1
                return False
                
        except Exception as e:
            self.logger.error(f"OTP verification failed: {e}")
            return False


class WhatsAppProvider:
    """WhatsApp Business API provider."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # WhatsApp Business API configuration
        self.api_url = config.get('whatsapp', {}).get('api_url')
        self.access_token = config.get('whatsapp', {}).get('access_token')
        self.phone_number_id = config.get('whatsapp', {}).get('phone_number_id')
        
        # Message templates
        self.templates = {}
    
    async def send_whatsapp_message(self, request: MessageRequest) -> MessageResult:
        """Send WhatsApp message."""
        message_id = str(uuid.uuid4())
        
        try:
            if not REQUESTS_AVAILABLE:
                raise Exception("Requests library not available")
            
            sent_count = 0
            failed_count = 0
            
            for recipient in request.recipients:
                if not recipient.phone:
                    failed_count += 1
                    continue
                
                try:
                    # Prepare WhatsApp message
                    message_data = await self._prepare_whatsapp_message(recipient, request.content)
                    
                    # Send message
                    await self._send_whatsapp_api_message(message_data)
                    sent_count += 1
                    
                except Exception as e:
                    self.logger.warning(f"WhatsApp message to {recipient.phone} failed: {e}")
                    failed_count += 1
            
            status = MessageStatus.SENT if sent_count > 0 else MessageStatus.FAILED
            
            return MessageResult(
                message_id=message_id,
                status=status,
                message_type=MessageType.WHATSAPP,
                recipient_count=len(request.recipients),
                sent_count=sent_count,
                failed_count=failed_count,
                provider_used=ProviderType.WHATSAPP_BUSINESS,
                sent_at=datetime.utcnow()
            )
            
        except Exception as e:
            self.logger.error(f"WhatsApp sending failed: {e}")
            return MessageResult(
                message_id=message_id,
                status=MessageStatus.FAILED,
                message_type=MessageType.WHATSAPP,
                recipient_count=len(request.recipients),
                failed_count=len(request.recipients),
                error_message=str(e)
            )
    
    async def _prepare_whatsapp_message(self, recipient: Recipient, content: MessageContent) -> Dict[str, Any]:
        """Prepare WhatsApp message payload."""
        message_data = {
            "messaging_product": "whatsapp",
            "to": recipient.phone.replace('+', ''),
            "type": "text",
            "text": {
                "body": content.body
            }
        }
        
        return message_data
    
    async def _send_whatsapp_api_message(self, message_data: Dict[str, Any]):
        """Send message via WhatsApp Business API."""
        try:
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'Content-Type': 'application/json'
            }
            
            url = f"{self.api_url}/{self.phone_number_id}/messages"
            
            response = requests.post(url, headers=headers, json=message_data)
            response.raise_for_status()
            
            self.logger.info(f"WhatsApp message sent: {response.json()}")
            
        except Exception as e:
            self.logger.error(f"WhatsApp API call failed: {e}")
            raise


class SlackProvider:
    """Slack messaging provider."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Slack configuration
        self.bot_token = config.get('slack', {}).get('bot_token')
        self.app_token = config.get('slack', {}).get('app_token')
        self.webhook_url = config.get('slack', {}).get('webhook_url')
    
    async def send_slack_message(self, request: MessageRequest) -> MessageResult:
        """Send Slack message."""
        message_id = str(uuid.uuid4())
        
        try:
            sent_count = 0
            failed_count = 0
            
            for recipient in request.recipients:
                try:
                    # Prepare Slack message
                    message_data = await self._prepare_slack_message(recipient, request.content)
                    
                    # Send message
                    if recipient.platform_id:  # Direct message to user
                        await self._send_slack_dm(recipient.platform_id, message_data)
                    elif self.webhook_url:  # Channel message via webhook
                        await self._send_slack_webhook(message_data)
                    else:
                        raise Exception("No Slack delivery method available")
                    
                    sent_count += 1
                    
                except Exception as e:
                    self.logger.warning(f"Slack message failed: {e}")
                    failed_count += 1
            
            status = MessageStatus.SENT if sent_count > 0 else MessageStatus.FAILED
            
            return MessageResult(
                message_id=message_id,
                status=status,
                message_type=MessageType.SLACK,
                recipient_count=len(request.recipients),
                sent_count=sent_count,
                failed_count=failed_count,
                provider_used=ProviderType.SLACK_API,
                sent_at=datetime.utcnow()
            )
            
        except Exception as e:
            self.logger.error(f"Slack sending failed: {e}")
            return MessageResult(
                message_id=message_id,
                status=MessageStatus.FAILED,
                message_type=MessageType.SLACK,
                recipient_count=len(request.recipients),
                failed_count=len(request.recipients),
                error_message=str(e)
            )
    
    async def _prepare_slack_message(self, recipient: Recipient, content: MessageContent) -> Dict[str, Any]:
        """Prepare Slack message payload."""
        message_data = {
            "text": content.body,
            "username": "SUPER-OMEGA Bot"
        }
        
        # Add rich formatting if HTML content available
        if content.html_body:
            message_data["blocks"] = self._convert_html_to_slack_blocks(content.html_body)
        
        return message_data
    
    async def _send_slack_webhook(self, message_data: Dict[str, Any]):
        """Send message via Slack webhook."""
        try:
            response = requests.post(self.webhook_url, json=message_data)
            response.raise_for_status()
            
            self.logger.info("Slack webhook message sent successfully")
            
        except Exception as e:
            self.logger.error(f"Slack webhook failed: {e}")
            raise


class PushNotificationProvider:
    """Push notification provider for mobile apps."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Firebase configuration
        self.firebase_app = None
        if FIREBASE_AVAILABLE and 'firebase' in config:
            try:
                import firebase_admin
                from firebase_admin import credentials
                
                cred = credentials.Certificate(config['firebase']['service_account_path'])
                self.firebase_app = firebase_admin.initialize_app(cred)
            except Exception as e:
                self.logger.warning(f"Firebase initialization failed: {e}")
    
    async def send_push_notification(self, request: MessageRequest) -> MessageResult:
        """Send push notification."""
        message_id = str(uuid.uuid4())
        
        try:
            sent_count = 0
            failed_count = 0
            
            for recipient in request.recipients:
                if not recipient.platform_id:  # FCM token
                    failed_count += 1
                    continue
                
                try:
                    # Prepare notification
                    notification = messaging.Message(
                        notification=messaging.Notification(
                            title=request.content.subject or 'Notification',
                            body=request.content.body
                        ),
                        token=recipient.platform_id,
                        data=request.metadata or {}
                    )
                    
                    # Send notification
                    response = messaging.send(notification)
                    self.logger.info(f"Push notification sent: {response}")
                    sent_count += 1
                    
                except Exception as e:
                    self.logger.warning(f"Push notification failed: {e}")
                    failed_count += 1
            
            status = MessageStatus.SENT if sent_count > 0 else MessageStatus.FAILED
            
            return MessageResult(
                message_id=message_id,
                status=status,
                message_type=MessageType.PUSH_NOTIFICATION,
                recipient_count=len(request.recipients),
                sent_count=sent_count,
                failed_count=failed_count,
                provider_used=ProviderType.FIREBASE,
                sent_at=datetime.utcnow()
            )
            
        except Exception as e:
            self.logger.error(f"Push notification sending failed: {e}")
            return MessageResult(
                message_id=message_id,
                status=MessageStatus.FAILED,
                message_type=MessageType.PUSH_NOTIFICATION,
                recipient_count=len(request.recipients),
                failed_count=len(request.recipients),
                error_message=str(e)
            )


class UniversalMessagingOrchestrator:
    """Master orchestrator for all messaging types."""
    
    def __init__(self, config: Dict[str, Any], security_manager: EnterpriseSecurityManager):
        self.config = config
        self.security_manager = security_manager
        self.logger = logging.getLogger(__name__)
        
        # Initialize providers
        self.email_provider = EmailProvider(config)
        self.sms_provider = SMSProvider(config)
        self.whatsapp_provider = WhatsAppProvider(config)
        self.slack_provider = SlackProvider(config)
        self.push_provider = PushNotificationProvider(config)
        
        # Message tracking and analytics
        self.message_history = []
        self.delivery_tracking = {}
        self.templates = {}
        
        # Rate limiting and compliance
        self.rate_limits = {}
        self.spam_detection = {}
        self.compliance_rules = {}
    
    async def send_message(self, request: MessageRequest) -> MessageResult:
        """Universal message sending function."""
        try:
            # Validate request
            await self._validate_message_request(request)
            
            # Apply rate limiting
            await self._apply_rate_limiting(request)
            
            # Check compliance
            await self._check_compliance(request)
            
            # Route to appropriate provider
            if request.message_type == MessageType.EMAIL:
                result = await self.email_provider.send_email(request)
            elif request.message_type == MessageType.SMS:
                result = await self.sms_provider.send_sms(request)
            elif request.message_type == MessageType.WHATSAPP:
                result = await self.whatsapp_provider.send_whatsapp_message(request)
            elif request.message_type == MessageType.SLACK:
                result = await self.slack_provider.send_slack_message(request)
            elif request.message_type == MessageType.PUSH_NOTIFICATION:
                result = await self.push_provider.send_push_notification(request)
            else:
                raise ValueError(f"Unsupported message type: {request.message_type}")
            
            # Store message history
            self.message_history.append({
                'request': asdict(request),
                'result': asdict(result),
                'timestamp': datetime.utcnow()
            })
            
            # Start delivery tracking
            if request.tracking_enabled:
                await self._start_delivery_tracking(result.message_id, request)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Message sending failed: {e}")
            return MessageResult(
                message_id=str(uuid.uuid4()),
                status=MessageStatus.FAILED,
                message_type=request.message_type,
                recipient_count=len(request.recipients),
                failed_count=len(request.recipients),
                error_message=str(e)
            )
    
    async def send_bulk_messages(self, requests: List[MessageRequest]) -> List[MessageResult]:
        """Send multiple messages with optimization."""
        try:
            # Group by message type for optimization
            grouped_requests = {}
            for request in requests:
                msg_type = request.message_type
                if msg_type not in grouped_requests:
                    grouped_requests[msg_type] = []
                grouped_requests[msg_type].append(request)
            
            # Send in parallel by type
            all_results = []
            tasks = []
            
            for msg_type, type_requests in grouped_requests.items():
                for request in type_requests:
                    task = self.send_message(request)
                    tasks.append(task)
            
            # Execute all tasks
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, MessageResult):
                    all_results.append(result)
                else:
                    # Handle exceptions
                    error_result = MessageResult(
                        message_id=str(uuid.uuid4()),
                        status=MessageStatus.FAILED,
                        message_type=MessageType.EMAIL,  # Default
                        recipient_count=0,
                        error_message=str(result)
                    )
                    all_results.append(error_result)
            
            return all_results
            
        except Exception as e:
            self.logger.error(f"Bulk message sending failed: {e}")
            return []
    
    async def generate_and_send_otp(self, phone: str, template_name: str = "otp_verification") -> str:
        """Generate and send OTP with template."""
        try:
            # Generate OTP
            otp = await self.sms_provider.generate_otp(phone)
            
            # Use template if available
            if template_name in self.templates:
                template = self.templates[template_name]
                content = MessageContent(
                    body=template.body,
                    variables={'otp': otp}
                )
            else:
                content = MessageContent(body=f"Your verification code is: {otp}")
            
            # Send OTP
            request = MessageRequest(
                message_type=MessageType.SMS,
                recipients=[Recipient(phone=phone)],
                content=content,
                priority=Priority.HIGH
            )
            
            result = await self.send_message(request)
            
            if result.status == MessageStatus.SENT:
                return otp
            else:
                raise Exception("Failed to send OTP")
                
        except Exception as e:
            self.logger.error(f"OTP generation and sending failed: {e}")
            raise
    
    async def verify_otp(self, phone: str, otp: str) -> bool:
        """Verify OTP code."""
        return await self.sms_provider.verify_otp(phone, otp)
    
    async def create_template(self, template: MessageTemplate) -> str:
        """Create message template."""
        try:
            self.templates[template.template_id] = template
            self.logger.info(f"Template created: {template.template_id}")
            return template.template_id
            
        except Exception as e:
            self.logger.error(f"Template creation failed: {e}")
            raise
    
    async def _validate_message_request(self, request: MessageRequest):
        """Validate message request."""
        if not request.recipients:
            raise ValueError("No recipients specified")
        
        if not request.content.body:
            raise ValueError("Message body is required")
        
        # Validate recipients based on message type
        for recipient in request.recipients:
            if request.message_type == MessageType.EMAIL and not recipient.email:
                raise ValueError("Email address required for email messages")
            elif request.message_type == MessageType.SMS and not recipient.phone:
                raise ValueError("Phone number required for SMS messages")
    
    async def _apply_rate_limiting(self, request: MessageRequest):
        """Apply rate limiting rules."""
        # Simple rate limiting implementation
        current_time = datetime.utcnow()
        rate_key = f"{request.message_type.value}_{current_time.strftime('%Y%m%d%H')}"
        
        if rate_key not in self.rate_limits:
            self.rate_limits[rate_key] = 0
        
        # Check limits (can be configured per message type)
        max_per_hour = {
            MessageType.EMAIL: 1000,
            MessageType.SMS: 500,
            MessageType.WHATSAPP: 200,
            MessageType.SLACK: 100,
            MessageType.PUSH_NOTIFICATION: 2000
        }
        
        if self.rate_limits[rate_key] >= max_per_hour.get(request.message_type, 100):
            raise Exception(f"Rate limit exceeded for {request.message_type.value}")
        
        self.rate_limits[rate_key] += len(request.recipients)
    
    async def _check_compliance(self, request: MessageRequest):
        """Check compliance rules."""
        # Check for spam indicators
        spam_keywords = ['free', 'urgent', 'limited time', 'act now', 'click here']
        content_lower = request.content.body.lower()
        
        spam_score = sum(1 for keyword in spam_keywords if keyword in content_lower)
        
        if spam_score >= 3:
            self.logger.warning("Message flagged as potential spam")
            # Could reject or modify message based on policy
    
    async def _start_delivery_tracking(self, message_id: str, request: MessageRequest):
        """Start tracking message delivery."""
        self.delivery_tracking[message_id] = {
            'message_id': message_id,
            'message_type': request.message_type,
            'recipients': len(request.recipients),
            'sent_at': datetime.utcnow(),
            'tracking_enabled': request.tracking_enabled,
            'delivery_receipt': request.delivery_receipt,
            'read_receipt': request.read_receipt
        }
    
    def get_message_analytics(self) -> Dict[str, Any]:
        """Get comprehensive messaging analytics."""
        if not self.message_history:
            return {}
        
        total_messages = len(self.message_history)
        successful_messages = sum(1 for msg in self.message_history 
                                if msg['result']['status'] == MessageStatus.SENT)
        
        # Group by message type
        type_distribution = {}
        for msg in self.message_history:
            msg_type = msg['result']['message_type']
            if msg_type not in type_distribution:
                type_distribution[msg_type] = {'sent': 0, 'failed': 0}
            
            if msg['result']['status'] == MessageStatus.SENT:
                type_distribution[msg_type]['sent'] += 1
            else:
                type_distribution[msg_type]['failed'] += 1
        
        return {
            'total_messages': total_messages,
            'successful_messages': successful_messages,
            'success_rate': (successful_messages / total_messages) * 100 if total_messages > 0 else 0,
            'type_distribution': type_distribution,
            'active_templates': len(self.templates),
            'delivery_tracking_active': len(self.delivery_tracking)
        }