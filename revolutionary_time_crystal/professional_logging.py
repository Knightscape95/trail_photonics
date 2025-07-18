"""
Professional Logging System for Time-Crystal Pipeline
====================================================

Critical fix for code review recommendations: Strip emojis and banners for 
production; switch to `logging` with DEBUG/INFO/WARN levels.

This module provides:
- Structured logging with proper levels
- Professional formatting without emojis
- Configurable output destinations
- Performance logging
- Audit trail for reproducibility

Author: Revolutionary Time-Crystal Team
Date: July 2025
Status: Code Review Fix - Logging Overhaul
"""

import logging
import logging.handlers
import sys
import os
import json
import time
from typing import Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
import warnings

# Suppress emoji and banner usage warning
warnings.filterwarnings('ignore', category=UserWarning, message='.*emoji.*')


@dataclass
class LoggingConfig:
    """Configuration for logging system."""
    
    # Log levels
    console_level: str = "INFO"
    file_level: str = "DEBUG"
    
    # Output destinations
    log_file: Optional[str] = None
    console_output: bool = True
    
    # Formatting
    include_timestamps: bool = True
    include_process_info: bool = True
    include_module_info: bool = True
    
    # Rotation (for file logging)
    max_log_size_mb: int = 100
    backup_count: int = 5
    
    # Performance logging
    enable_performance_logging: bool = True
    performance_threshold_seconds: float = 1.0


class ProfessionalFormatter(logging.Formatter):
    """Professional formatter without emojis or excessive decoration."""
    
    def __init__(self, 
                 include_timestamps: bool = True,
                 include_process_info: bool = True,
                 include_module_info: bool = True):
        """
        Initialize professional formatter.
        
        Args:
            include_timestamps: Include timestamp in logs
            include_process_info: Include process/thread info
            include_module_info: Include module name
        """
        
        # Build format string
        fmt_parts = []
        
        if include_timestamps:
            fmt_parts.append("%(asctime)s")
        
        if include_process_info:
            fmt_parts.append("[PID:%(process)d]")
        
        fmt_parts.extend([
            "%(levelname)-8s",
        ])
        
        if include_module_info:
            fmt_parts.append("%(name)s")
        
        fmt_parts.append("%(message)s")
        
        format_string = " - ".join(fmt_parts)
        
        super().__init__(
            fmt=format_string,
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with emoji removal."""
        
        # Clean message of emojis and excessive decoration
        original_msg = record.getMessage()
        cleaned_msg = self._clean_message(original_msg)
        
        # Temporarily replace message
        record.msg = cleaned_msg
        record.args = ()
        
        formatted = super().format(record)
        
        # Restore original message
        record.msg = original_msg
        
        return formatted
    
    def _clean_message(self, message: str) -> str:
        """
        Clean message of emojis and excessive decoration while preserving LaTeX math.
        
        Args:
            message: Original log message
        
        Returns:
            Cleaned message with LaTeX math preserved
        """
        
        # FIX: Preserve LaTeX math expressions before cleaning
        latex_patterns = []
        cleaned = message
        
        # Preserve inline LaTeX: $...$
        import re
        inline_math_pattern = r'\$[^$]+\$'
        inline_matches = re.findall(inline_math_pattern, message)
        for i, match in enumerate(inline_matches):
            placeholder = f"__LATEX_INLINE_{i}__"
            latex_patterns.append((placeholder, match))
            cleaned = cleaned.replace(match, placeholder)
        
        # Preserve display LaTeX: \[...\] and \(...\)
        display_patterns = [
            (r'\\\\?\[([^]]+)\\\\?\]', r'\\[..\\]'),
            (r'\\\\?\(([^)]+)\\\\?\)', r'\\(..\\)'),
            (r'\\begin\{[^}]+\}.*?\\end\{[^}]+\}', r'\\begin{..}..\\end{..}')
        ]
        
        for pattern, replacement in display_patterns:
            display_matches = re.findall(pattern, cleaned, re.DOTALL)
            for i, match in enumerate(display_matches):
                full_match = re.search(pattern, cleaned, re.DOTALL)
                if full_match:
                    placeholder = f"__LATEX_DISPLAY_{len(latex_patterns)}__"
                    latex_patterns.append((placeholder, full_match.group(0)))
                    cleaned = cleaned.replace(full_match.group(0), placeholder)
        
        # Preserve equations with = signs in scientific context
        equation_pattern = r'[A-Za-z0-9_]+\s*=\s*[A-Za-z0-9_+\-*/().]+(?:\s*[A-Za-z]+)?'
        equation_matches = re.findall(equation_pattern, cleaned)
        for i, match in enumerate(equation_matches):
            placeholder = f"__EQUATION_{i}__"
            latex_patterns.append((placeholder, match))
            cleaned = cleaned.replace(match, placeholder)
        
        # Remove common emojis (unchanged)
        # Remove common emojis (unchanged)
        emoji_replacements = {
            '🚀': '[START]',
            '✅': '[SUCCESS]',
            '❌': '[ERROR]',
            '⚠️': '[WARNING]',
            '🔍': '[CHECK]',
            '📊': '[DATA]',
            '🎯': '[TARGET]',
            '🔧': '[CONFIG]',
            '📈': '[PROGRESS]',
            '🎉': '[COMPLETE]',
            '💾': '[MEMORY]',
            '🧪': '[TEST]',
            '📁': '[FILE]',
            '🌟': '[RESULT]',
            '💡': '[INFO]',
            '🔬': '[SCIENCE]',
            '⚡': '[FAST]',
            '🎮': '[CONTROL]',
            '🔥': '[HOT]',
            '🚨': '[ALERT]',
            '📋': '[LIST]',
            '🎭': '[MOCK]',
            '⏱️': '[TIME]',
            '🏁': '[FINISH]',
            '🔄': '[PROCESS]',
            '🎪': '[EVENT]',
            '🧹': '[CLEANUP]',
            '⭐': '[STAR]',
            '🌊': '[WAVE]',
            '🔮': '[PREDICT]',
            '🎵': '[MUSIC]',
            '🎨': '[ART]'
        }
        
        # Apply emoji cleaning
        for emoji, replacement in emoji_replacements.items():
            cleaned = cleaned.replace(emoji, replacement)
        
        # Remove excessive ASCII decoration
        cleaned = self._remove_ascii_banners(cleaned)
        
        # Clean up extra whitespace
        cleaned = ' '.join(cleaned.split())
        
        # FIX: Restore LaTeX expressions after cleaning
        for placeholder, original in latex_patterns:
            cleaned = cleaned.replace(placeholder, original)
        
        return cleaned
    
    def _remove_ascii_banners(self, message: str) -> str:
        """Remove ASCII banners and excessive decoration."""
        
        lines = message.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip lines that are mostly decoration
            if self._is_decoration_line(line):
                continue
            
            # Clean up remaining decoration
            cleaned_line = line.strip('=').strip('-').strip('*').strip('#').strip()
            
            if cleaned_line:  # Only keep non-empty lines
                cleaned_lines.append(cleaned_line)
        
        return ' '.join(cleaned_lines)
    
    def _is_decoration_line(self, line: str) -> bool:
        """Check if line is mostly decoration."""
        
        # Remove whitespace
        stripped = line.strip()
        
        if not stripped:
            return True
        
        # Check if line is mostly decoration characters
        decoration_chars = set('=-*#~^+|\\/<>{}[]()_`.')
        char_count = len(stripped)
        decoration_count = sum(1 for c in stripped if c in decoration_chars)
        
        # If more than 80% decoration, consider it a banner
        return decoration_count / char_count > 0.8


class ProfessionalLogger:
    """
    Professional logging wrapper with enhanced functionality.
    
    Provides a high-level interface for professional logging with:
    - Multiple log levels (debug, info, warning, error, critical)
    - Performance timing
    - Context management
    - Audit trail support
    """
    
    def __init__(self, name: str, config: Optional[LoggingConfig] = None):
        """
        Initialize professional logger.
        
        Args:
            name: Logger name
            config: Logging configuration (uses default if None)
        """
        self.name = name
        self.config = config or LoggingConfig()
        
        # Set up underlying logger
        self.logger = logging.getLogger(name)
        self._setup_logger()
        
        # Initialize specialized loggers (will be set after class definitions)
        self._perf_logger = None
        self._audit_logger = None
    
    def _setup_logger(self):
        """Set up the underlying logger with proper configuration."""
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Set log level
        self.logger.setLevel(getattr(logging, self.config.file_level.upper()))
        
        # Create formatter
        formatter = ProfessionalFormatter(
            include_timestamps=self.config.include_timestamps,
            include_process_info=self.config.include_process_info,
            include_module_info=self.config.include_module_info
        )
        
        # Console handler
        if self.config.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, self.config.console_level.upper()))
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if self.config.log_file:
            # Ensure log directory exists
            log_path = Path(self.config.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            if self.config.max_log_size_mb > 0:
                # Rotating file handler
                file_handler = logging.handlers.RotatingFileHandler(
                    self.config.log_file,
                    maxBytes=self.config.max_log_size_mb * 1024 * 1024,
                    backupCount=self.config.backup_count
                )
            else:
                # Regular file handler
                file_handler = logging.FileHandler(self.config.log_file)
            
            file_handler.setLevel(getattr(logging, self.config.file_level.upper()))
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    # Standard logging methods
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self.logger.critical(message, **kwargs)
    
    # Performance logging methods
    @property
    def perf_logger(self):
        """Lazy initialization of performance logger."""
        if self._perf_logger is None:
            self._perf_logger = PerformanceLogger(
                self.logger, 
                threshold_seconds=self.config.performance_threshold_seconds
            )
        return self._perf_logger
    
    @property
    def audit_logger(self):
        """Lazy initialization of audit logger."""
        if self._audit_logger is None:
            self._audit_logger = AuditLogger(self.logger)
        return self._audit_logger
    
    def start_timer(self, operation: str):
        """Start timing an operation."""
        self.perf_logger.start_timer(operation)
    
    def end_timer(self, operation: str, metadata: Optional[Dict[str, Any]] = None) -> float:
        """End timing an operation and return elapsed time."""
        return self.perf_logger.end_timer(operation, metadata)
    
    def time_operation(self, operation: str):
        """Context manager for timing operations."""
        return self.perf_logger.time_operation(operation)
    
    # Audit logging methods
    def log_parameter_change(self, parameter: str, old_value: Any, new_value: Any, reason: str):
        """Log parameter change for audit trail."""
        self.audit_logger.log_parameter_change(parameter, old_value, new_value, reason)
    
    def log_seed_usage(self, seed: int, context: str):
        """Log random seed usage for reproducibility."""
        self.audit_logger.log_seed_usage(seed, context)
    
    def export_audit_trail(self, filepath: str):
        """Export audit trail to file."""
        self.audit_logger.export_audit_trail(filepath)
    
    # FIX: Added CI integration capabilities
    def export_ci_metrics(self, output_path: str = "ci_metrics.json") -> Dict[str, Any]:
        """
        Export metrics for CI integration.
        
        Args:
            output_path: Path to export CI metrics JSON
            
        Returns:
            Dictionary of CI metrics
        """
        import json
        from pathlib import Path
        
        ci_metrics = {
            'timestamp': time.time(),
            'logger_name': self.name,
            'performance_metrics': self._get_performance_summary(),
            'error_counts': self._get_error_summary(),
            'audit_trail_summary': self._get_audit_summary()
        }
        
        # Export to file for CI consumption
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(ci_metrics, f, indent=2, default=str)
        
        return ci_metrics
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary for CI."""
        if self._perf_logger:
            return {
                'total_operations': len(self._perf_logger.timers),
                'active_timers': len([t for t in self._perf_logger.timers.values() if t.get('start_time')]),
                'avg_operation_time': getattr(self._perf_logger, 'avg_time', 0.0)
            }
        return {'performance_logging_disabled': True}
    
    def _get_error_summary(self) -> Dict[str, int]:
        """Get error count summary for CI."""
        # This would ideally track error counts, simplified for now
        return {
            'total_errors': 0,
            'critical_errors': 0,
            'warnings': 0
        }
    
    def _get_audit_summary(self) -> Dict[str, Any]:
        """Get audit trail summary for CI."""
        if self._audit_logger:
            return {
                'parameter_changes': getattr(self._audit_logger, 'change_count', 0),
                'seed_operations': getattr(self._audit_logger, 'seed_count', 0),
                'audit_enabled': True
            }
        return {'audit_logging_disabled': True}
    
    # Context management
    def get_child(self, suffix: str) -> 'ProfessionalLogger':
        """Get child logger with suffix."""
        child_name = f"{self.name}.{suffix}"
        return ProfessionalLogger(child_name, self.config)


class PerformanceLogger:
    """Logger for performance metrics and timing."""
    
    def __init__(self, logger: logging.Logger, threshold_seconds: float = 1.0):
        """
        Initialize performance logger.
        
        Args:
            logger: Base logger instance
            threshold_seconds: Minimum time to log
        """
        self.logger = logger
        self.threshold = threshold_seconds
        self.timers = {}
    
    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        self.timers[operation] = time.time()
        self.logger.debug(f"Starting operation: {operation}")
    
    def end_timer(self, operation: str, **metadata) -> float:
        """
        End timing an operation.
        
        Args:
            operation: Operation name
            **metadata: Additional metadata to log
        
        Returns:
            Elapsed time in seconds
        """
        
        if operation not in self.timers:
            self.logger.warning(f"Timer for operation '{operation}' not found")
            return 0.0
        
        elapsed = time.time() - self.timers[operation]
        del self.timers[operation]
        
        # Log if above threshold
        if elapsed >= self.threshold:
            metadata_str = ""
            if metadata:
                metadata_str = f" ({', '.join(f'{k}={v}' for k, v in metadata.items())})"
            
            self.logger.info(f"Operation '{operation}' completed in {elapsed:.2f}s{metadata_str}")
        else:
            self.logger.debug(f"Operation '{operation}' completed in {elapsed:.3f}s")
        
        return elapsed
    
    def time_operation(self, operation: str):
        """
        Context manager for timing operations.
        
        Args:
            operation: Operation name
        """
        return TimingContext(self, operation)


class TimingContext:
    """Context manager for operation timing."""
    
    def __init__(self, perf_logger: PerformanceLogger, operation: str):
        """
        Initialize timing context.
        
        Args:
            perf_logger: Performance logger instance
            operation: Operation name
        """
        self.perf_logger = perf_logger
        self.operation = operation
    
    def __enter__(self):
        """Enter timing context."""
        self.perf_logger.start_timer(self.operation)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit timing context."""
        self.perf_logger.end_timer(self.operation)


class AuditLogger:
    """Logger for audit trail and reproducibility."""
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize audit logger.
        
        Args:
            logger: Base logger instance
        """
        self.logger = logger
        self.audit_trail = []
    
    def log_parameter_change(self, 
                           parameter: str, 
                           old_value: Any, 
                           new_value: Any,
                           reason: Optional[str] = None) -> None:
        """
        Log parameter changes for reproducibility.
        
        Args:
            parameter: Parameter name
            old_value: Previous value
            new_value: New value
            reason: Reason for change
        """
        
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'parameter': parameter,
            'old_value': str(old_value),
            'new_value': str(new_value),
            'reason': reason
        }
        
        self.audit_trail.append(audit_entry)
        
        reason_str = f" (reason: {reason})" if reason else ""
        self.logger.info(f"Parameter changed: {parameter} = {old_value} -> {new_value}{reason_str}")
    
    def log_seed_usage(self, seed: int, context: str) -> None:
        """
        Log random seed usage.
        
        Args:
            seed: Random seed value
            context: Context where seed is used
        """
        
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': 'seed_usage',
            'seed': seed,
            'context': context
        }
        
        self.audit_trail.append(audit_entry)
        self.logger.info(f"Random seed used: {seed} in context '{context}'")
    
    def export_audit_trail(self, filepath: str) -> None:
        """
        Export audit trail to file.
        
        Args:
            filepath: Output file path
        """
        
        with open(filepath, 'w') as f:
            json.dump(self.audit_trail, f, indent=2)
        
        self.logger.info(f"Audit trail exported to: {filepath}")


def setup_logging(config: LoggingConfig) -> Dict[str, logging.Logger]:
    """
    Set up professional logging system.
    
    Args:
        config: Logging configuration
    
    Returns:
        Dictionary of configured loggers
    """
    
    # Create formatters
    console_formatter = ProfessionalFormatter(
        include_timestamps=True,
        include_process_info=False,  # Less verbose for console
        include_module_info=True
    )
    
    file_formatter = ProfessionalFormatter(
        include_timestamps=True,
        include_process_info=True,
        include_module_info=True
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Allow all levels, filter at handler level
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    if config.console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, config.console_level))
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # File handler (with rotation)
    if config.log_file:
        log_path = Path(config.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            config.log_file,
            maxBytes=config.max_log_size_mb * 1024 * 1024,
            backupCount=config.backup_count
        )
        file_handler.setLevel(getattr(logging, config.file_level))
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Create specialized loggers
    loggers = {
        'main': logging.getLogger('time_crystal'),
        'performance': logging.getLogger('time_crystal.performance'),
        'audit': logging.getLogger('time_crystal.audit'),
        'memory': logging.getLogger('time_crystal.memory'),
        'concurrency': logging.getLogger('time_crystal.concurrency'),
        'scientific': logging.getLogger('time_crystal.scientific')
    }
    
    # Add performance and audit capabilities
    perf_logger = PerformanceLogger(
        loggers['performance'], 
        config.performance_threshold_seconds
    )
    
    audit_logger = AuditLogger(loggers['audit'])
    
    # Store additional capabilities
    loggers['perf'] = perf_logger
    loggers['audit_manager'] = audit_logger
    
    return loggers


def migrate_legacy_logging(source_file: str, 
                         target_file: str,
                         dry_run: bool = True) -> Dict[str, int]:
    """
    Migrate legacy emoji/banner logging to professional format.
    
    Args:
        source_file: Source Python file to migrate
        target_file: Target file for migrated code
        dry_run: If True, only analyze without writing
    
    Returns:
        Migration statistics
    """
    
    if not os.path.exists(source_file):
        raise FileNotFoundError(f"Source file not found: {source_file}")
    
    with open(source_file, 'r') as f:
        content = f.read()
    
    # Track migration statistics
    stats = {
        'print_statements': 0,
        'emoji_occurrences': 0,
        'ascii_banners': 0,
        'lines_changed': 0
    }
    
    lines = content.split('\n')
    migrated_lines = []
    
    for line_num, line in enumerate(lines):
        original_line = line
        modified = False
        
        # Replace print statements with logging
        if 'print(' in line and not line.strip().startswith('#'):
            # Simple print to logging conversion
            indentation = line[:len(line) - len(line.lstrip())]
            
            if any(emoji in line for emoji in ['🚀', '✅', '❌', '⚠️']):
                # Extract message content
                if '🚀' in line or '[START]' in line:
                    line = f"{indentation}logger.info(\"Starting: {line.strip()}\")"
                elif '✅' in line or '[SUCCESS]' in line:
                    line = f"{indentation}logger.info(\"Success: {line.strip()}\")"
                elif '❌' in line or '[ERROR]' in line:
                    line = f"{indentation}logger.error(\"Error: {line.strip()}\")"
                elif '⚠️' in line or '[WARNING]' in line:
                    line = f"{indentation}logger.warning(\"Warning: {line.strip()}\")"
                else:
                    line = f"{indentation}logger.info({line.strip()[6:-1]})"  # Remove print()
                
                modified = True
                stats['print_statements'] += 1
        
        # Count emojis
        emoji_count = sum(line.count(emoji) for emoji in ['🚀', '✅', '❌', '⚠️', '🔍', '📊'])
        if emoji_count > 0:
            stats['emoji_occurrences'] += emoji_count
        
        # Detect ASCII banners
        if len(line.strip()) > 20 and line.count('=') > 10:
            stats['ascii_banners'] += 1
            # Convert to simple comment
            line = f"# {line.strip('= ')}"
            modified = True
        
        if modified:
            stats['lines_changed'] += 1
        
        migrated_lines.append(line)
    
    # Write migrated content
    if not dry_run:
        os.makedirs(os.path.dirname(target_file), exist_ok=True)
        with open(target_file, 'w') as f:
            f.write('\n'.join(migrated_lines))
    
    return stats


def create_logging_config_file(filepath: str, config: LoggingConfig) -> None:
    """
    Create logging configuration file.
    
    Args:
        filepath: Path to configuration file
        config: Logging configuration
    """
    
    config_dict = {
        'console_level': config.console_level,
        'file_level': config.file_level,
        'log_file': config.log_file,
        'console_output': config.console_output,
        'include_timestamps': config.include_timestamps,
        'include_process_info': config.include_process_info,
        'include_module_info': config.include_module_info,
        'max_log_size_mb': config.max_log_size_mb,
        'backup_count': config.backup_count,
        'enable_performance_logging': config.enable_performance_logging,
        'performance_threshold_seconds': config.performance_threshold_seconds
    }
    
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=2)


if __name__ == "__main__":
    # Test professional logging system
    print("Testing Professional Logging System")
    
    # Create test configuration
    config = LoggingConfig(
        console_level="INFO",
        file_level="DEBUG",
        log_file="test_logs/time_crystal.log",
        performance_threshold_seconds=0.1
    )
    
    # Set up logging
    loggers = setup_logging(config)
    
    main_logger = loggers['main']
    perf_logger = loggers['perf']
    audit_logger = loggers['audit_manager']
    
    # Test various log levels
    main_logger.debug("Debug message - should only appear in file")
    main_logger.info("Info message - appears in console and file")
    main_logger.warning("Warning message")
    main_logger.error("Error message")
    
    # Test performance logging
    with perf_logger.time_operation("test_operation"):
        time.sleep(0.15)  # Simulate work
    
    # Test audit logging
    audit_logger.log_parameter_change("resolution", 30, 25, "memory optimization")
    audit_logger.log_seed_usage(42, "global_initialization")
    
    # Export audit trail
    audit_logger.export_audit_trail("test_logs/audit_trail.json")
    
    print("Professional logging test completed")
    print("Check test_logs/ directory for output files")
