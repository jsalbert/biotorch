import os
import json

from jsonschema import validators, Draft7Validator


def read_schema(schema_name):
    with open(os.path.normpath(os.path.join(
            os.path.dirname(__file__), '..', 'schemas',
            schema_name + '.json'
    ))) as schema:
        return json.load(schema)


def validate_config(instance, schema_name, defaults=True):
    # Validate `instance` with `schema_name` schema values.
    # If defaults set to True, `instance` will be initialized with default values from `schema`.
    with open(os.path.normpath(os.path.join(
            os.path.dirname(__file__), '..', 'schemas',
            schema_name + '.json'
    ))) as schema:
        if defaults:
            default_validator = extend_schema_with_default(Draft7Validator)
            try:
                default_validator(json.load(schema)).validate(instance)
            except ValueError:
                raise ValueError("Error when validating the default schema.")
        else:
            try:
                jsonschema.validate(instance, json.load(schema))
            except ValueError:
                raise ValueError("Error when validating the schema.")


def extend_schema_with_default(validator_class):
    validate_properties = validator_class.VALIDATORS["properties"]

    def set_defaults(validator, properties, instance, schema):
        for property_, subschema in properties.items():
            if "default" in subschema and not isinstance(instance, list):
                instance.setdefault(property_, subschema["default"])

        for error in validate_properties(
                validator, properties, instance, schema,
        ):
            yield error

    return validators.extend(
        validator_class, {"properties": set_defaults},
    )
