import os
from typing import Optional
from unittest.mock import patch

import pytest
from langchain_core.runnables import RunnableConfig
from layout_prompter.utils.configuration import Configuration
from pydantic import Field


class MockConfiguration(Configuration):
    """Mock configuration class for testing purposes."""

    test_field: Optional[str] = Field(default=None)
    another_field: Optional[int] = Field(default=None)
    required_field: str = Field(default="default_value")


def test_from_runnable_config_empty_config():
    """Test creating Configuration from empty RunnableConfig."""
    config = MockConfiguration.from_runnable_config(None)

    assert isinstance(config, MockConfiguration)
    assert config.test_field is None
    assert config.another_field is None
    assert config.required_field == "default_value"


def test_from_runnable_config_with_configurable():
    """Test creating Configuration from RunnableConfig with configurable values."""
    runnable_config: RunnableConfig = {
        "configurable": {
            "test_field": "config_value",
            "another_field": 42,
            "required_field": "overridden_value",
        }
    }

    config = MockConfiguration.from_runnable_config(runnable_config)

    assert config.test_field == "config_value"
    assert config.another_field == 42
    assert config.required_field == "overridden_value"


def test_from_runnable_config_empty_configurable():
    """Test creating Configuration from RunnableConfig with empty configurable."""
    runnable_config: RunnableConfig = {"configurable": {}}

    config = MockConfiguration.from_runnable_config(runnable_config)

    assert config.test_field is None
    assert config.another_field is None
    assert config.required_field == "default_value"


def test_from_runnable_config_missing_configurable_key():
    """Test creating Configuration from RunnableConfig without configurable key."""
    runnable_config: RunnableConfig = {"other_key": "other_value"}

    config = MockConfiguration.from_runnable_config(runnable_config)

    assert config.test_field is None
    assert config.another_field is None
    assert config.required_field == "default_value"


@patch.dict(os.environ, {"TEST_FIELD": "env_value", "ANOTHER_FIELD": "123"})
def test_from_runnable_config_with_environment_variables():
    """Test that environment variables are used when available."""
    config = MockConfiguration.from_runnable_config(None)

    assert config.test_field == "env_value"
    assert (
        config.another_field == 123
    )  # Pydantic converts string to int for the field type
    assert config.required_field == "default_value"


@patch.dict(os.environ, {"TEST_FIELD": "env_value"})
def test_from_runnable_config_env_overrides_configurable():
    """Test that environment variables take precedence over configurable values."""
    runnable_config: RunnableConfig = {
        "configurable": {"test_field": "config_value", "another_field": 42}
    }

    config = MockConfiguration.from_runnable_config(runnable_config)

    # Environment variable should take precedence
    assert config.test_field == "env_value"
    # Configurable value should be used when no env var exists
    assert config.another_field == 42


@patch.dict(os.environ, {"REQUIRED_FIELD": "env_required"})
def test_from_runnable_config_env_overrides_default():
    """Test that environment variables override default values."""
    config = MockConfiguration.from_runnable_config(None)

    assert config.required_field == "env_required"


def test_from_runnable_config_ignores_none_values():
    """Test that None values are ignored and not passed to the constructor."""
    runnable_config: RunnableConfig = {
        "configurable": {"test_field": None, "another_field": 42}
    }

    config = MockConfiguration.from_runnable_config(runnable_config)

    # None values should be ignored, so default should be used
    assert config.test_field is None  # This is the default None value
    assert config.another_field == 42


@patch.dict(os.environ, {"TEST_FIELD": ""})
def test_from_runnable_config_empty_string_env_var():
    """Test behavior with empty string environment variable."""
    config = MockConfiguration.from_runnable_config(None)

    # Empty string should be treated as a value (not None)
    assert config.test_field == ""


def test_from_runnable_config_partial_fields():
    """Test with only some fields provided in configurable."""
    runnable_config: RunnableConfig = {
        "configurable": {
            "test_field": "partial_config"
            # another_field and required_field not provided
        }
    }

    config = MockConfiguration.from_runnable_config(runnable_config)

    assert config.test_field == "partial_config"
    assert config.another_field is None
    assert config.required_field == "default_value"


def test_from_runnable_config_unknown_fields_ignored():
    """Test that unknown fields in configurable are ignored."""
    runnable_config: RunnableConfig = {
        "configurable": {
            "test_field": "known_field",
            "unknown_field": "should_be_ignored",
        }
    }

    config = MockConfiguration.from_runnable_config(runnable_config)

    assert config.test_field == "known_field"
    assert not hasattr(config, "unknown_field")


@patch.dict(os.environ, {"TEST_FIELD": "env_val", "UNKNOWN_ENV": "ignored"})
def test_from_runnable_config_unknown_env_vars_ignored():
    """Test that environment variables for unknown fields are ignored."""
    config = MockConfiguration.from_runnable_config(None)

    assert config.test_field == "env_val"
    assert not hasattr(config, "unknown_env")


def test_configuration_model_config():
    """Test that Configuration has correct model config."""
    config = Configuration()
    assert config.model_config["arbitrary_types_allowed"] is True


def test_configuration_inheritance():
    """Test that Configuration can be inherited."""
    config = MockConfiguration()
    assert isinstance(config, Configuration)
    assert isinstance(config, MockConfiguration)


def test_pydantic_validation_with_wrong_type():
    """Test that Pydantic validation works correctly."""
    runnable_config: RunnableConfig = {
        "configurable": {
            "another_field": "not_an_int"  # Should be int but providing string
        }
    }

    # This should raise a validation error since Pydantic can't convert the string to int
    with pytest.raises(Exception):  # Could be ValidationError or similar
        MockConfiguration.from_runnable_config(runnable_config)


@patch.dict(os.environ, {}, clear=True)
def test_from_runnable_config_clean_environment():
    """Test behavior with completely clean environment."""
    runnable_config: RunnableConfig = {"configurable": {"test_field": "only_config"}}

    config = MockConfiguration.from_runnable_config(runnable_config)

    assert config.test_field == "only_config"
    assert config.another_field is None
    assert config.required_field == "default_value"
