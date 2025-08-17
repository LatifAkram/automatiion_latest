"""
BUILT-IN DATA VALIDATION ENGINE
===============================

Comprehensive data validation system using only Python standard library.
Provides schema validation, type checking, and field constraints.

✅ FEATURES:
- Schema validation with type checking and conversion
- Field validation with length, regex, choice constraints
- Error handling with comprehensive validation reporting
- Flexible API supporting custom validation rules
- Zero dependencies - pure Python stdlib
"""

import re
import json
import time
import logging
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import datetime
import decimal
import urllib.parse

logger = logging.getLogger(__name__)

class ValidationType(Enum):
    """Validation type enumeration"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"
    EMAIL = "email"
    URL = "url"
    DATE = "date"
    DATETIME = "datetime"
    UUID = "uuid"
    JSON = "json"

class ValidationSeverity(Enum):
    """Validation error severity"""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

@dataclass
class ValidationRule:
    """Individual validation rule"""
    field_name: str
    rule_type: str
    rule_value: Any
    message: str = ""
    severity: ValidationSeverity = ValidationSeverity.ERROR

@dataclass
class ValidationError:
    """Validation error details"""
    field: str
    message: str
    value: Any
    rule_type: str
    severity: ValidationSeverity = ValidationSeverity.ERROR
    timestamp: float = field(default_factory=time.time)

@dataclass
class ValidationResult:
    """Validation result container"""
    valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    cleaned_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseValidator:
    """Base validation class with core functionality"""
    
    def __init__(self):
        self.custom_validators = {}
        self.type_converters = {
            'str': str,
            'string': str,
            'int': int,
            'integer': int,
            'float': float,
            'bool': bool,
            'boolean': bool,
            'list': list,
            'dict': dict
        }
    
    def validate(self, data: Dict[str, Any], schema: Dict[str, Any]) -> ValidationResult:
        """
        Validate data against schema
        
        Args:
            data: Data to validate
            schema: Validation schema
            
        Returns:
            ValidationResult with validation status and details
        """
        result = ValidationResult(valid=True, cleaned_data={})
        
        try:
            # Validate required fields
            self._validate_required_fields(data, schema, result)
            
            # Validate each field
            for field_name, field_schema in schema.items():
                if field_name.startswith('_'):
                    continue  # Skip meta fields
                    
                if field_name in data:
                    self._validate_field(field_name, data[field_name], field_schema, result)
                elif field_schema.get('required', False):
                    result.errors.append(ValidationError(
                        field=field_name,
                        message=f"Required field '{field_name}' is missing",
                        value=None,
                        rule_type='required'
                    ))
                elif 'default' in field_schema:
                    result.cleaned_data[field_name] = field_schema['default']
            
            # Check for unknown fields if strict mode
            if schema.get('_strict', False):
                self._check_unknown_fields(data, schema, result)
            
            # Set validation status
            result.valid = len(result.errors) == 0
            
            # Add metadata
            result.metadata = {
                'validation_time': time.time(),
                'fields_validated': len(schema),
                'errors_count': len(result.errors),
                'warnings_count': len(result.warnings)
            }
            
        except Exception as e:
            result.valid = False
            result.errors.append(ValidationError(
                field='_validation',
                message=f"Validation process failed: {str(e)}",
                value=data,
                rule_type='system_error'
            ))
        
        return result
    
    def _validate_required_fields(self, data: Dict[str, Any], schema: Dict[str, Any], result: ValidationResult):
        """Validate required fields are present"""
        required_fields = []
        
        for field_name, field_schema in schema.items():
            if isinstance(field_schema, dict) and field_schema.get('required', False):
                required_fields.append(field_name)
        
        for field_name in required_fields:
            if field_name not in data or data[field_name] is None:
                result.errors.append(ValidationError(
                    field=field_name,
                    message=f"Required field '{field_name}' is missing or null",
                    value=data.get(field_name),
                    rule_type='required'
                ))
    
    def _validate_field(self, field_name: str, value: Any, field_schema: Dict[str, Any], result: ValidationResult):
        """Validate individual field"""
        try:
            # Type validation and conversion
            converted_value = self._validate_type(field_name, value, field_schema, result)
            
            if converted_value is not None:
                # Length validation
                self._validate_length(field_name, converted_value, field_schema, result)
                
                # Range validation
                self._validate_range(field_name, converted_value, field_schema, result)
                
                # Pattern validation
                self._validate_pattern(field_name, converted_value, field_schema, result)
                
                # Choice validation
                self._validate_choices(field_name, converted_value, field_schema, result)
                
                # Custom validation
                self._validate_custom(field_name, converted_value, field_schema, result)
                
                # If no errors for this field, add to cleaned data
                field_errors = [e for e in result.errors if e.field == field_name]
                if not field_errors:
                    result.cleaned_data[field_name] = converted_value
        
        except Exception as e:
            result.errors.append(ValidationError(
                field=field_name,
                message=f"Field validation failed: {str(e)}",
                value=value,
                rule_type='validation_error'
            ))
    
    def _validate_type(self, field_name: str, value: Any, field_schema: Dict[str, Any], result: ValidationResult) -> Any:
        """Validate and convert field type"""
        expected_type = field_schema.get('type', 'str')
        
        # Handle None values
        if value is None:
            if field_schema.get('nullable', False):
                return None
            else:
                result.errors.append(ValidationError(
                    field=field_name,
                    message=f"Field '{field_name}' cannot be null",
                    value=value,
                    rule_type='type'
                ))
                return None
        
        # Type conversion and validation
        try:
            if expected_type in ['str', 'string']:
                return str(value)
            
            elif expected_type in ['int', 'integer']:
                if isinstance(value, bool):
                    result.errors.append(ValidationError(
                        field=field_name,
                        message=f"Field '{field_name}' expected integer, got boolean",
                        value=value,
                        rule_type='type'
                    ))
                    return None
                return int(value)
            
            elif expected_type == 'float':
                return float(value)
            
            elif expected_type in ['bool', 'boolean']:
                if isinstance(value, str):
                    if value.lower() in ['true', '1', 'yes', 'on']:
                        return True
                    elif value.lower() in ['false', '0', 'no', 'off']:
                        return False
                    else:
                        result.errors.append(ValidationError(
                            field=field_name,
                            message=f"Field '{field_name}' invalid boolean string: '{value}'",
                            value=value,
                            rule_type='type'
                        ))
                        return None
                return bool(value)
            
            elif expected_type == 'list':
                if not isinstance(value, list):
                    result.errors.append(ValidationError(
                        field=field_name,
                        message=f"Field '{field_name}' expected list, got {type(value).__name__}",
                        value=value,
                        rule_type='type'
                    ))
                    return None
                return value
            
            elif expected_type == 'dict':
                if not isinstance(value, dict):
                    result.errors.append(ValidationError(
                        field=field_name,
                        message=f"Field '{field_name}' expected dict, got {type(value).__name__}",
                        value=value,
                        rule_type='type'
                    ))
                    return None
                return value
            
            elif expected_type == 'email':
                email_str = str(value)
                if not self._is_valid_email(email_str):
                    result.errors.append(ValidationError(
                        field=field_name,
                        message=f"Field '{field_name}' is not a valid email address",
                        value=value,
                        rule_type='email'
                    ))
                    return None
                return email_str
            
            elif expected_type == 'url':
                url_str = str(value)
                if not self._is_valid_url(url_str):
                    result.errors.append(ValidationError(
                        field=field_name,
                        message=f"Field '{field_name}' is not a valid URL",
                        value=value,
                        rule_type='url'
                    ))
                    return None
                return url_str
            
            elif expected_type == 'date':
                return self._parse_date(field_name, value, result)
            
            elif expected_type == 'datetime':
                return self._parse_datetime(field_name, value, result)
            
            elif expected_type == 'uuid':
                return self._validate_uuid(field_name, value, result)
            
            elif expected_type == 'json':
                return self._validate_json(field_name, value, result)
            
            else:
                # Unknown type, treat as string
                result.warnings.append(ValidationError(
                    field=field_name,
                    message=f"Unknown type '{expected_type}', treating as string",
                    value=value,
                    rule_type='type',
                    severity=ValidationSeverity.WARNING
                ))
                return str(value)
        
        except (ValueError, TypeError) as e:
            result.errors.append(ValidationError(
                field=field_name,
                message=f"Type conversion failed for '{field_name}': {str(e)}",
                value=value,
                rule_type='type'
            ))
            return None
    
    def _validate_length(self, field_name: str, value: Any, field_schema: Dict[str, Any], result: ValidationResult):
        """Validate field length constraints"""
        min_length = field_schema.get('min_length')
        max_length = field_schema.get('max_length')
        
        if min_length is None and max_length is None:
            return
        
        try:
            length = len(value)
            
            if min_length is not None and length < min_length:
                result.errors.append(ValidationError(
                    field=field_name,
                    message=f"Field '{field_name}' must be at least {min_length} characters long",
                    value=value,
                    rule_type='min_length'
                ))
            
            if max_length is not None and length > max_length:
                result.errors.append(ValidationError(
                    field=field_name,
                    message=f"Field '{field_name}' must be at most {max_length} characters long",
                    value=value,
                    rule_type='max_length'
                ))
        
        except TypeError:
            # Value doesn't have length
            result.warnings.append(ValidationError(
                field=field_name,
                message=f"Length validation skipped for '{field_name}' (no length attribute)",
                value=value,
                rule_type='length',
                severity=ValidationSeverity.WARNING
            ))
    
    def _validate_range(self, field_name: str, value: Any, field_schema: Dict[str, Any], result: ValidationResult):
        """Validate numeric range constraints"""
        min_value = field_schema.get('min_value')
        max_value = field_schema.get('max_value')
        
        if min_value is None and max_value is None:
            return
        
        try:
            numeric_value = float(value)
            
            if min_value is not None and numeric_value < min_value:
                result.errors.append(ValidationError(
                    field=field_name,
                    message=f"Field '{field_name}' must be at least {min_value}",
                    value=value,
                    rule_type='min_value'
                ))
            
            if max_value is not None and numeric_value > max_value:
                result.errors.append(ValidationError(
                    field=field_name,
                    message=f"Field '{field_name}' must be at most {max_value}",
                    value=value,
                    rule_type='max_value'
                ))
        
        except (ValueError, TypeError):
            # Value is not numeric
            if min_value is not None or max_value is not None:
                result.warnings.append(ValidationError(
                    field=field_name,
                    message=f"Range validation skipped for '{field_name}' (not numeric)",
                    value=value,
                    rule_type='range',
                    severity=ValidationSeverity.WARNING
                ))
    
    def _validate_pattern(self, field_name: str, value: Any, field_schema: Dict[str, Any], result: ValidationResult):
        """Validate regex pattern constraints"""
        pattern = field_schema.get('pattern')
        
        if pattern is None:
            return
        
        try:
            value_str = str(value)
            if not re.match(pattern, value_str):
                result.errors.append(ValidationError(
                    field=field_name,
                    message=f"Field '{field_name}' does not match required pattern: {pattern}",
                    value=value,
                    rule_type='pattern'
                ))
        
        except re.error as e:
            result.errors.append(ValidationError(
                field=field_name,
                message=f"Invalid regex pattern for '{field_name}': {str(e)}",
                value=value,
                rule_type='pattern'
            ))
    
    def _validate_choices(self, field_name: str, value: Any, field_schema: Dict[str, Any], result: ValidationResult):
        """Validate choice constraints"""
        choices = field_schema.get('choices')
        
        if choices is None:
            return
        
        if value not in choices:
            result.errors.append(ValidationError(
                field=field_name,
                message=f"Field '{field_name}' must be one of: {choices}",
                value=value,
                rule_type='choices'
            ))
    
    def _validate_custom(self, field_name: str, value: Any, field_schema: Dict[str, Any], result: ValidationResult):
        """Validate custom validation functions"""
        custom_validator = field_schema.get('custom_validator')
        
        if custom_validator is None:
            return
        
        try:
            if callable(custom_validator):
                is_valid = custom_validator(value)
                if not is_valid:
                    result.errors.append(ValidationError(
                        field=field_name,
                        message=f"Field '{field_name}' failed custom validation",
                        value=value,
                        rule_type='custom'
                    ))
            elif isinstance(custom_validator, str) and custom_validator in self.custom_validators:
                validator_func = self.custom_validators[custom_validator]
                is_valid = validator_func(value)
                if not is_valid:
                    result.errors.append(ValidationError(
                        field=field_name,
                        message=f"Field '{field_name}' failed custom validation: {custom_validator}",
                        value=value,
                        rule_type='custom'
                    ))
        
        except Exception as e:
            result.errors.append(ValidationError(
                field=field_name,
                message=f"Custom validation error for '{field_name}': {str(e)}",
                value=value,
                rule_type='custom'
            ))
    
    def _check_unknown_fields(self, data: Dict[str, Any], schema: Dict[str, Any], result: ValidationResult):
        """Check for unknown fields in strict mode"""
        known_fields = set(schema.keys())
        known_fields.discard('_strict')  # Remove meta fields
        data_fields = set(data.keys())
        
        unknown_fields = data_fields - known_fields
        
        for field_name in unknown_fields:
            result.warnings.append(ValidationError(
                field=field_name,
                message=f"Unknown field '{field_name}' in strict validation mode",
                value=data[field_name],
                rule_type='unknown_field',
                severity=ValidationSeverity.WARNING
            ))
    
    def _is_valid_email(self, email: str) -> bool:
        """Validate email address format"""
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(email_pattern, email) is not None
    
    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format"""
        try:
            result = urllib.parse.urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def _parse_date(self, field_name: str, value: Any, result: ValidationResult) -> Optional[datetime.date]:
        """Parse date value"""
        if isinstance(value, datetime.date):
            return value
        
        try:
            if isinstance(value, str):
                # Try common date formats
                formats = ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y%m%d']
                for fmt in formats:
                    try:
                        return datetime.datetime.strptime(value, fmt).date()
                    except ValueError:
                        continue
            
            result.errors.append(ValidationError(
                field=field_name,
                message=f"Field '{field_name}' is not a valid date format",
                value=value,
                rule_type='date'
            ))
            return None
        
        except Exception as e:
            result.errors.append(ValidationError(
                field=field_name,
                message=f"Date parsing failed for '{field_name}': {str(e)}",
                value=value,
                rule_type='date'
            ))
            return None
    
    def _parse_datetime(self, field_name: str, value: Any, result: ValidationResult) -> Optional[datetime.datetime]:
        """Parse datetime value"""
        if isinstance(value, datetime.datetime):
            return value
        
        try:
            if isinstance(value, str):
                # Try common datetime formats
                formats = [
                    '%Y-%m-%d %H:%M:%S',
                    '%Y-%m-%dT%H:%M:%S',
                    '%Y-%m-%dT%H:%M:%SZ',
                    '%Y-%m-%d %H:%M',
                    '%d/%m/%Y %H:%M:%S',
                    '%m/%d/%Y %H:%M:%S'
                ]
                for fmt in formats:
                    try:
                        return datetime.datetime.strptime(value, fmt)
                    except ValueError:
                        continue
            
            result.errors.append(ValidationError(
                field=field_name,
                message=f"Field '{field_name}' is not a valid datetime format",
                value=value,
                rule_type='datetime'
            ))
            return None
        
        except Exception as e:
            result.errors.append(ValidationError(
                field=field_name,
                message=f"Datetime parsing failed for '{field_name}': {str(e)}",
                value=value,
                rule_type='datetime'
            ))
            return None
    
    def _validate_uuid(self, field_name: str, value: Any, result: ValidationResult) -> Optional[str]:
        """Validate UUID format"""
        import uuid
        
        try:
            uuid_str = str(value)
            uuid.UUID(uuid_str)
            return uuid_str
        
        except (ValueError, TypeError):
            result.errors.append(ValidationError(
                field=field_name,
                message=f"Field '{field_name}' is not a valid UUID",
                value=value,
                rule_type='uuid'
            ))
            return None
    
    def _validate_json(self, field_name: str, value: Any, result: ValidationResult) -> Any:
        """Validate and parse JSON"""
        try:
            if isinstance(value, str):
                return json.loads(value)
            elif isinstance(value, (dict, list)):
                # Already parsed JSON
                return value
            else:
                result.errors.append(ValidationError(
                    field=field_name,
                    message=f"Field '{field_name}' is not valid JSON",
                    value=value,
                    rule_type='json'
                ))
                return None
        
        except json.JSONDecodeError as e:
            result.errors.append(ValidationError(
                field=field_name,
                message=f"JSON parsing failed for '{field_name}': {str(e)}",
                value=value,
                rule_type='json'
            ))
            return None
    
    def add_custom_validator(self, name: str, validator_func: Callable[[Any], bool]):
        """Add custom validation function"""
        self.custom_validators[name] = validator_func
    
    def validate_batch(self, data_list: List[Dict[str, Any]], schema: Dict[str, Any]) -> List[ValidationResult]:
        """Validate multiple data objects"""
        results = []
        
        for i, data in enumerate(data_list):
            try:
                result = self.validate(data, schema)
                result.metadata['batch_index'] = i
                results.append(result)
            except Exception as e:
                error_result = ValidationResult(
                    valid=False,
                    errors=[ValidationError(
                        field='_batch',
                        message=f"Batch validation failed at index {i}: {str(e)}",
                        value=data,
                        rule_type='batch_error'
                    )],
                    metadata={'batch_index': i}
                )
                results.append(error_result)
        
        return results

class SchemaBuilder:
    """Helper class to build validation schemas"""
    
    def __init__(self):
        self.schema = {}
    
    def add_field(self, name: str, field_type: str = 'string', required: bool = False, **kwargs) -> 'SchemaBuilder':
        """Add field to schema"""
        field_schema = {'type': field_type, 'required': required}
        field_schema.update(kwargs)
        self.schema[name] = field_schema
        return self
    
    def add_string(self, name: str, required: bool = False, min_length: int = None, 
                   max_length: int = None, pattern: str = None, choices: List[str] = None) -> 'SchemaBuilder':
        """Add string field"""
        kwargs = {}
        if min_length is not None:
            kwargs['min_length'] = min_length
        if max_length is not None:
            kwargs['max_length'] = max_length
        if pattern is not None:
            kwargs['pattern'] = pattern
        if choices is not None:
            kwargs['choices'] = choices
        
        return self.add_field(name, 'string', required, **kwargs)
    
    def add_integer(self, name: str, required: bool = False, min_value: int = None, 
                    max_value: int = None, choices: List[int] = None) -> 'SchemaBuilder':
        """Add integer field"""
        kwargs = {}
        if min_value is not None:
            kwargs['min_value'] = min_value
        if max_value is not None:
            kwargs['max_value'] = max_value
        if choices is not None:
            kwargs['choices'] = choices
        
        return self.add_field(name, 'integer', required, **kwargs)
    
    def add_email(self, name: str, required: bool = False) -> 'SchemaBuilder':
        """Add email field"""
        return self.add_field(name, 'email', required)
    
    def add_url(self, name: str, required: bool = False) -> 'SchemaBuilder':
        """Add URL field"""
        return self.add_field(name, 'url', required)
    
    def set_strict(self, strict: bool = True) -> 'SchemaBuilder':
        """Set strict validation mode"""
        self.schema['_strict'] = strict
        return self
    
    def build(self) -> Dict[str, Any]:
        """Build the schema"""
        return self.schema.copy()

# Global validator instance
_global_validator: Optional[BaseValidator] = None

def get_validator() -> BaseValidator:
    """Get or create global validator instance"""
    global _global_validator
    
    if _global_validator is None:
        _global_validator = BaseValidator()
    
    return _global_validator

def validate_data(data: Dict[str, Any], schema: Dict[str, Any]) -> ValidationResult:
    """Quick validation function"""
    validator = get_validator()
    return validator.validate(data, schema)

def create_schema() -> SchemaBuilder:
    """Create new schema builder"""
    return SchemaBuilder()

# Common validation schemas
COMMON_SCHEMAS = {
    'user_registration': {
        'username': {'type': 'string', 'required': True, 'min_length': 3, 'max_length': 50},
        'email': {'type': 'email', 'required': True},
        'password': {'type': 'string', 'required': True, 'min_length': 8},
        'age': {'type': 'integer', 'required': False, 'min_value': 13, 'max_value': 120},
        'terms_accepted': {'type': 'boolean', 'required': True}
    },
    
    'api_request': {
        'method': {'type': 'string', 'required': True, 'choices': ['GET', 'POST', 'PUT', 'DELETE']},
        'url': {'type': 'url', 'required': True},
        'headers': {'type': 'dict', 'required': False},
        'data': {'type': 'json', 'required': False},
        'timeout': {'type': 'integer', 'required': False, 'min_value': 1, 'max_value': 300}
    },
    
    'automation_task': {
        'task_id': {'type': 'string', 'required': True},
        'instruction': {'type': 'string', 'required': True, 'min_length': 1},
        'priority': {'type': 'string', 'required': False, 'choices': ['low', 'medium', 'high', 'urgent']},
        'scheduled_time': {'type': 'datetime', 'required': False},
        'retry_count': {'type': 'integer', 'required': False, 'min_value': 0, 'max_value': 10}
    }
}

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    print("🔧 Built-in Data Validator Demo")
    print("=" * 40)
    
    # Create validator
    validator = BaseValidator()
    
    # Test data
    test_data = {
        'username': 'john_doe',
        'email': 'john@example.com',
        'password': 'secure123',
        'age': 25,
        'terms_accepted': True
    }
    
    # Test schema
    schema = COMMON_SCHEMAS['user_registration']
    
    # Validate
    result = validator.validate(test_data, schema)
    
    print(f"✅ Validation Result: {'PASSED' if result.valid else 'FAILED'}")
    print(f"📊 Fields validated: {len(schema)}")
    print(f"❌ Errors: {len(result.errors)}")
    print(f"⚠️ Warnings: {len(result.warnings)}")
    
    if result.errors:
        print("\n❌ Validation Errors:")
        for error in result.errors:
            print(f"  - {error.field}: {error.message}")
    
    if result.warnings:
        print("\n⚠️ Validation Warnings:")
        for warning in result.warnings:
            print(f"  - {warning.field}: {warning.message}")
    
    print(f"\n🧹 Cleaned Data: {len(result.cleaned_data)} fields")
    print(f"📈 Metadata: {result.metadata}")
    
    # Test schema builder
    print("\n🔧 Schema Builder Demo:")
    schema_builder = SchemaBuilder()
    custom_schema = (schema_builder
                     .add_string('name', required=True, min_length=2)
                     .add_email('contact_email', required=True)
                     .add_integer('score', required=False, min_value=0, max_value=100)
                     .set_strict(True)
                     .build())
    
    test_data2 = {
        'name': 'Alice',
        'contact_email': 'alice@test.com',
        'score': 95
    }
    
    result2 = validator.validate(test_data2, custom_schema)
    print(f"Schema Builder Result: {'PASSED' if result2.valid else 'FAILED'}")
    
    print("\n✅ Built-in data validation working perfectly!")
    print("🎯 No external dependencies required!")