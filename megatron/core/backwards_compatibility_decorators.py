"""
Backward Compatibility Decorators

This module provides decorators for managing backward compatibility
in Megatron Core's public API.

Usage:
    from megatron.core.backwards_compatibility_decorators import exempt_from_compat_check, deprecated

    # Mark function as exempt from compatibility checks
    @exempt_from_compat_check
    def experimental_feature():
        pass

    # Mark function as deprecated
    @deprecated(version="1.0.0", removal_version="2.0.0", alternative="new_function")
    def old_function():
        pass
"""

import functools
import warnings
from typing import Optional, Callable


def exempt_from_compat_check(func: Callable) -> Callable:
    """
    Mark a function as exempt from backward compatibility checks.
    
    Use this decorator for:
    - Internal APIs not intended for external use
    - Experimental features that may change
    - Functions explicitly documented as unstable
    
    Args:
        func: The function to mark as exempt
    
    Returns:
        The original function with an exemption marker
    
    Example:
        @exempt_from_compat_check
        def experimental_api():
            '''This API may change without notice'''
            pass
    """
    func._exempt_from_compat_check = True
    return func


def deprecated(
    version: str,
    removal_version: Optional[str] = None,
    alternative: Optional[str] = None,
    reason: Optional[str] = None
) -> Callable:
    """
    Mark a function as deprecated.
    
    This decorator:
    1. Adds deprecation metadata to the function
    2. Issues a DeprecationWarning when the function is called
    3. Allows the compatibility checker to track deprecation lifecycle
    
    Args:
        version: Version where deprecation starts (e.g., "1.0.0")
        removal_version: Version where function will be removed (e.g., "2.0.0")
        alternative: Name of the recommended replacement function
        reason: Optional explanation for the deprecation
    
    Returns:
        Decorator function
    
    Example:
        @deprecated(
            version="1.0.0",
            removal_version="2.0.0",
            alternative="new_train_model",
            reason="Improved performance and cleaner API"
        )
        def old_train_model(config):
            pass
    """
    def decorator(func: Callable) -> Callable:
        # Add metadata
        func._deprecated = True
        func._deprecated_version = version
        func._removal_version = removal_version
        func._alternative = alternative
        func._deprecation_reason = reason
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Build warning message
            msg_parts = [f"{func.__name__} is deprecated since version {version}."]
            
            if alternative:
                msg_parts.append(f"Use {alternative} instead.")
            
            if removal_version:
                msg_parts.append(f"Will be removed in version {removal_version}.")
            
            if reason:
                msg_parts.append(f"Reason: {reason}")
            
            warnings.warn(
                " ".join(msg_parts),
                DeprecationWarning,
                stacklevel=2
            )
            
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def internal_api(func: Callable) -> Callable:
    """
    Mark a function as internal API (not for external use).
    
    This is semantically similar to exempt_from_compat_check but
    more explicitly communicates that the function is internal.
    
    Args:
        func: The function to mark as internal
    
    Returns:
        The original function with an internal API marker
    
    Example:
        @internal_api
        def _internal_helper():
            '''For internal use only'''
            pass
    """
    func._internal_api = True
    func._exempt_from_compat_check = True  # Also exempt from checks
    return func

