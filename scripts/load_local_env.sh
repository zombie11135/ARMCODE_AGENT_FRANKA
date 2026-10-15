#!/usr/bin/env bash
# Sourced by launchers. This is a trusted, machine-local shell configuration.
ARM_AGENT_LOCAL_ENV="${ARM_AGENT_LOCAL_ENV:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/config/local.env}"
if [[ -f "$ARM_AGENT_LOCAL_ENV" ]]; then
  source "$ARM_AGENT_LOCAL_ENV"
fi
