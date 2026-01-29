import inspect
from typing import Any

import pytest

from megatron.core.typed_torch import copy_signature, not_none


def source_func(a: int, *, b: str) -> str:
    """Sample function to copy the signature from."""
    return str(a) + b


class SourceClass:
    """Sample class with a method to copy the signature from."""

    def method(self, a: int, *, b: str) -> str:
        """Sample method to copy the signature from."""
        return str(a) + b


@copy_signature(source_func)
def dest_func_from_func(*args: Any, **kwargs: Any) -> list[str]:
    """Function with copied signature from source_func."""
    return [source_func(*args, **kwargs)]


@copy_signature(source_func, handle_return_type='overwrite')
def dest_func_from_func_overwrite(*args: Any, **kwargs: Any) -> object:
    """Function with copied signature from source_func, but overwritten return type."""
    return source_func(*args, **kwargs)


@copy_signature(SourceClass.method, handle_first_src_param='skip')
def dest_func_from_method(*args: Any, **kwargs: Any) -> int:
    """Function with copied signature from SourceClass.method."""
    return len(SourceClass().method(*args, **kwargs))


@copy_signature(SourceClass.method, handle_return_type='overwrite', handle_first_src_param='skip')
def dest_func_from_method_overwrite(*args: Any, **kwargs: Any) -> object:
    """Function with copied signature from SourceClass.method, but overwritten return type."""
    return SourceClass().method(*args, **kwargs)


class DestClass:
    """Class with methods that have copied signatures."""

    @copy_signature(source_func, handle_first_dst_param='preserve')
    def dest_method_from_func(self, *args: Any, **kwargs: Any) -> list[str]:
        """Method with copied signature from source_func."""
        return [source_func(*args, **kwargs)]

    @copy_signature(source_func, handle_return_type='overwrite', handle_first_dst_param='preserve')
    def dest_method_from_func_overwrite(self, *args: Any, **kwargs: Any) -> object:
        """Method with copied signature from source_func, but overwritten return type."""
        return source_func(*args, **kwargs)

    @classmethod
    @copy_signature(
        SourceClass.method, handle_first_src_param='skip', handle_first_dst_param='preserve'
    )
    def dest_method_from_method(cls, *args: Any, **kwargs: Any) -> int:
        """Class method with copied signature from SourceClass.method."""
        return len(SourceClass().method(*args, **kwargs))

    @copy_signature(
        SourceClass.method,
        handle_return_type='overwrite',
        handle_first_src_param='skip',
        handle_first_dst_param='preserve',
    )
    def dest_method_from_method_overwrite(self, *args: Any, **kwargs: Any) -> object:
        """Method with copied signature from SourceClass.method, but overwritten return type."""
        return SourceClass().method(*args, **kwargs)


class TestCopySignature:
    def test_original_return_type(self):
        """Test that the original return types are preserved."""
        f2f: list[str] = dest_func_from_func(1, b='a')
        assert f2f == ['1a']
        assert inspect.signature(dest_func_from_func) == inspect.Signature(
            [
                inspect.Parameter('a', inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=int),
                inspect.Parameter('b', inspect.Parameter.KEYWORD_ONLY, annotation=str),
            ],
            return_annotation=list[str],
        )

        m2f: int = dest_func_from_method(1, b='a')
        assert m2f == 2
        assert inspect.signature(dest_func_from_method) == inspect.Signature(
            [
                inspect.Parameter('a', inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=int),
                inspect.Parameter('b', inspect.Parameter.KEYWORD_ONLY, annotation=str),
            ],
            return_annotation=int,
        )

        f2m: list[str] = DestClass().dest_method_from_func(
            1, b='a'
        ) + DestClass.dest_method_from_func(DestClass(), 1, b='a')
        assert f2m == ['1a', '1a']
        assert inspect.signature(DestClass.dest_method_from_func) == inspect.Signature(
            [
                inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD),
                inspect.Parameter('a', inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=int),
                inspect.Parameter('b', inspect.Parameter.KEYWORD_ONLY, annotation=str),
            ],
            return_annotation=list[str],
        )

        m2m: int = DestClass.dest_method_from_method(1, b='a')
        assert m2m == 2
        assert inspect.signature(DestClass.dest_method_from_method) == inspect.Signature(
            [
                inspect.Parameter('a', inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=int),
                inspect.Parameter('b', inspect.Parameter.KEYWORD_ONLY, annotation=str),
            ],
            return_annotation=int,
        )

    def test_overwritten_return_type(self):
        """Test that the return types are overwritten correctly."""
        f2f: str = dest_func_from_func_overwrite(1, b='a')
        assert f2f == '1a'
        assert inspect.signature(dest_func_from_func_overwrite) == inspect.Signature(
            [
                inspect.Parameter('a', inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=int),
                inspect.Parameter('b', inspect.Parameter.KEYWORD_ONLY, annotation=str),
            ],
            return_annotation=str,
        )

        m2f: str = dest_func_from_method_overwrite(1, b='a')
        assert m2f == '1a'
        assert inspect.signature(dest_func_from_method_overwrite) == inspect.Signature(
            [
                inspect.Parameter('a', inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=int),
                inspect.Parameter('b', inspect.Parameter.KEYWORD_ONLY, annotation=str),
            ],
            return_annotation=str,
        )

        f2m: str = DestClass().dest_method_from_func_overwrite(
            1, b='a'
        ) + DestClass.dest_method_from_func_overwrite(DestClass(), 1, b='a')
        assert f2m == '1a1a'
        assert inspect.signature(DestClass.dest_method_from_func_overwrite) == inspect.Signature(
            [
                inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD),
                inspect.Parameter('a', inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=int),
                inspect.Parameter('b', inspect.Parameter.KEYWORD_ONLY, annotation=str),
            ],
            return_annotation=str,
        )

        m2m: str = DestClass().dest_method_from_method_overwrite(1, b='a')
        assert m2m == '1a'
        assert inspect.signature(DestClass.dest_method_from_method_overwrite) == inspect.Signature(
            [
                inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD),
                inspect.Parameter('a', inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=int),
                inspect.Parameter('b', inspect.Parameter.KEYWORD_ONLY, annotation=str),
            ],
            return_annotation=str,
        )


class TestNotNone:
    """Tests not_none."""

    def test_none(self):
        """Test that passing None raises a ValueError."""
        with pytest.raises(ValueError, match=r'Expected value to be not None'):
            not_none(None)

    def test_not_none(self):
        """Test that passing a non-None value returns the value."""
        value = 42
        result = not_none(value)
        assert result == value
