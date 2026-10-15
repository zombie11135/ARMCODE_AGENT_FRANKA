"""Implementations behind SimpleRobotActions, grouped by responsibility.

The facade owns shared task state and exposes the compatibility API. Modules
accept that facade explicitly, call its hooks for cross-module work, and import
model/device implementations only inside the operations that need them.
"""
