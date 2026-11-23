# Copyright (c) 2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/53c20f81cb575f19af34ff0ae1749fb13b07999b/tests/unit_tests/models/decorators/test_dispatch.py
# Source license: Apache-2.0.

"""Tests for megatron.bridge.models.decorators.dispatch module."""

import pytest

from megatron.bridge.models.decorators.dispatch import dispatch


class TestDispatch:
    """Test cases for dispatch decorator."""

    def test_basic_dispatch(self):
        """Test basic dispatch functionality with single types."""

        @dispatch
        def process(value):
            """Process a value based on its type."""
            pass

        @process.impl(int)
        def _process_int(value: int):
            return f"int: {value}"

        @process.impl(str)
        def _process_str(value: str):
            return f"str: {value}"

        assert process(42) == "int: 42"
        assert process("hello") == "str: hello"

    def test_multiple_type_registration(self):
        """Test registering one implementation for multiple types."""

        @dispatch
        def stringify(obj):
            """Convert object to string."""
            pass

        @stringify.impl(list)
        def _stringify_list(obj):
            return f"list: {', '.join(map(str, obj))}"

        @stringify.impl(set)
        def _stringify_set(obj):
            return f"set: {', '.join(map(str, sorted(obj)))}"

        assert stringify([1, 2, 3]) == "list: 1, 2, 3"
        assert stringify({4, 5, 6}) == "set: 4, 5, 6"

    def test_single_impl_multiple_types(self):
        """Test registering one implementation for multiple types using multiple decorators."""

        @dispatch
        def format_number(num):
            """Format number."""
            pass

        # Register same implementation for multiple types
        @format_number.impl(int, float)
        def _format_numeric(num):
            return f"number: {num}"

        assert format_number(42) == "number: 42"
        assert format_number(3.14) == "number: 3.14"

    def test_inheritance_dispatch(self):
        """Test dispatch with inheritance."""

        class Animal:
            pass

        class Dog(Animal):
            pass

        class Cat(Animal):
            pass

        @dispatch
        def make_sound(animal):
            """Make animal sound."""
            pass

        @make_sound.impl(Animal)
        def _sound_animal(animal: Animal):
            return "generic animal sound"

        @make_sound.impl(Dog)
        def _sound_dog(animal: Dog):
            return "woof"

        # Specific implementation takes precedence
        assert make_sound(Dog()) == "woof"

        # Falls back to parent class implementation
        assert make_sound(Cat()) == "generic animal sound"

    def test_tuple_dispatch(self):
        """Test dispatch with tuple keys."""

        @dispatch
        def combine(values):
            """Combine values based on their types."""
            pass

        @combine.impl((int, str))
        def _combine_int_str(values):
            num, text = values
            return f"{text} * {num} = " + (text * num)

        @combine.impl((str, str))
        def _combine_str_str(values):
            return " + ".join(values)

        assert combine((3, "hi")) == "hi * 3 = hihihi"
        assert combine(("hello", "world")) == "hello + world"

    def test_not_implemented_error(self):
        """Test error when no implementation exists."""

        @dispatch
        def process_data(data):
            """Process data."""
            pass

        @process_data.impl(int)
        def _process_int(data: int):
            return data * 2

        # Should work for int
        assert process_data(5) == 10

        # Should raise NotImplementedError for unregistered type
        with pytest.raises(NotImplementedError) as exc_info:
            process_data(3.14)

        error_msg = str(exc_info.value)
        assert "No implementation found for type 'float'" in error_msg
        assert "process_data" in error_msg

    def test_empty_impl_error(self):
        """Test error when .impl() is called without arguments."""

        @dispatch
        def my_func(x):
            pass

        with pytest.raises(ValueError) as exc_info:

            @my_func.impl()
            def _impl(x):
                pass

        assert "Missing argument to .impl()" in str(exc_info.value)

    def test_dispatch_cache_behavior(self):
        """Test that dispatch results are cached."""
        call_count = 0

        @dispatch
        def cached_func(value):
            pass

        @cached_func.impl(str)
        def _cached_str(value: str):
            nonlocal call_count
            call_count += 1
            return value.upper()

        # First call should increment counter
        result1 = cached_func("hello")
        assert result1 == "HELLO"
        assert call_count == 1

        # Second call with same type should use cache
        result2 = cached_func("world")
        assert result2 == "WORLD"
        assert call_count == 2  # Implementation is called, but dispatch lookup is cached

    def test_class_dispatch(self):
        """Test dispatch with custom classes."""

        class MyClass:
            name = "MyClass"

        class MyOtherClass:
            name = "MyOtherClass"

        @dispatch
        def get_class_name(obj):
            pass

        @get_class_name.impl(MyClass)
        def _get_myclass_name(obj: MyClass):
            return obj.name

        @get_class_name.impl(MyOtherClass)
        def _get_other_name(obj: MyOtherClass):
            return obj.name

        assert get_class_name(MyClass()) == "MyClass"
        assert get_class_name(MyOtherClass()) == "MyOtherClass"

    def test_dispatch_repr(self):
        """Test dispatch representation."""

        @dispatch
        def my_dispatch(x, y=10):
            """My dispatch function."""
            pass

        @my_dispatch.impl(int)
        def _int_impl(x: int, y=10):
            return x + y

        repr_str = repr(my_dispatch)
        assert "Dispatch(my_dispatch" in repr_str
        assert "(int):" in repr_str
        assert "_int_impl" in repr_str

    def test_string_single_arg_multiple_values(self):
        """Strings should not be cached by type; distinct names resolve separately."""

        @dispatch
        def arch_to_impl(arch):
            pass

        @arch_to_impl.impl("Dog")
        def _dog(_):
            return "woof"

        @arch_to_impl.impl("Cat")
        def _cat(_):
            return "meow"

        assert arch_to_impl("Dog") == "woof"
        assert arch_to_impl("Cat") == "meow"

    def test_string_single_arg_dispatch(self):
        """Test single-arg dispatch registered and called with string."""

        @dispatch
        def arch_to_impl(arch):
            pass

        @arch_to_impl.impl("Dog")
        def _dog_impl(_):
            return "woof"

        assert arch_to_impl("Dog") == "woof"

    def test_single_arg_name_fallback_type_call(self):
        """Test single-arg name fallback: register with string, call with type."""

        class Dog:
            pass

        @dispatch
        def arch_to_impl(arch):
            pass

        @arch_to_impl.impl("Dog")
        def _dog_impl(_):
            return "woof"

        assert arch_to_impl(Dog) == "woof"

    def test_single_arg_name_fallback_string_call(self):
        """Test single-arg name fallback: register with type, call with string."""

        class Cat:
            pass

        @dispatch
        def arch_to_impl(arch):
            pass

        @arch_to_impl.impl(Cat)
        def _cat_impl(_):
            return "meow"

        assert arch_to_impl("Cat") == "meow"

    def test_tuple_string_dispatch(self):
        """Test tuple dispatch with string and a type."""

        class Trainer:
            pass

        @dispatch
        def bridge_key(key):
            pass

        @bridge_key.impl(("Dog", Trainer))
        def _impl_tuple_str(values):
            return "trained-dog"

        assert bridge_key(("Dog", Trainer())) == "trained-dog"

    def test_tuple_name_fallback_type_call(self):
        """Test tuple name fallback: register with string, call with type."""

        class Trainer:
            pass

        class Dog:
            pass

        @dispatch
        def bridge_key(key):
            pass

        @bridge_key.impl(("Dog", Trainer))
        def _impl_tuple_str(values):
            return "trained-dog"

        assert bridge_key((Dog, Trainer())) == "trained-dog"

    def test_tuple_name_fallback_string_call(self):
        """Test tuple name fallback: register with type, call with string."""

        class Trainer:
            pass

        class Dog:
            pass

        @dispatch
        def bridge_key(key):
            pass

        @bridge_key.impl((Dog, Trainer))
        def _impl_tuple_type(values):
            return "trained-dog"

        assert bridge_key(("Dog", Trainer())) == "trained-dog"

    def test_tuple_no_match_raises(self):
        """Test tuple dispatch raises when types do not match."""

        class Trainer:
            pass

        @dispatch
        def bridge_key(key):
            pass

        @bridge_key.impl(("Dog", Trainer))
        def _impl(values):
            return "trained"

        with pytest.raises(NotImplementedError):
            bridge_key(("Cat", Trainer()))
