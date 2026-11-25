"""
API routes for configuration status and metadata.

This module provides endpoints for initialization status checks
and supported provider/database type listings.
"""

import asyncio
from copy import deepcopy
from typing import Any, Dict, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from datus.api import deps
from datus.api.deps import AppContextDep, ServiceDep
from datus.api.models.base_models import Result
from datus.api.models.config_models import DatasourceListData, DatasourceSummary
from datus.configuration.agent_config import _SAFE_NAME_RE, DbConfig, load_model_config
from datus.configuration.agent_config_loader import configuration_manager
from datus.configuration.project_config import (
    ProjectOverride,
    ProjectTarget,
    load_project_override,
    save_project_override,
)
from datus.models.base import LLMBaseModel
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["configuration"])


class UpdateDatasourcesRequest(BaseModel):
    """Full desired state for `services.datasources`.

    Any existing datasource key absent from `datasources` will be deleted.
    """

    datasources: Dict[str, Dict[str, Any]]


class UpdateModelsRequest(BaseModel):
    """Optional full-replace for `models` and/or update to `target`.

    At least one of `models` or `target` must be provided.
    """

    models: Optional[Dict[str, Dict[str, Any]]] = None
    target: Optional[str] = None


class ProbeModelRequest(BaseModel):
    """Single LLM model config dict — flat shape matching IModelInfo."""

    model_config = {"extra": "allow"}

    type: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None


class ProbeDatasourceRequest(BaseModel):
    """Single datasource config dict — flat shape matching IDatasourceConfig."""

    model_config = {"extra": "allow"}

    type: str


def _probe_llm_sync(payload: Dict[str, Any]) -> None:
    """Build a one-shot LLM client from a raw dict and send a tiny probe."""
    model_cfg = load_model_config(payload)
    model_class_name = LLMBaseModel.MODEL_TYPE_MAP.get(model_cfg.type)
    if model_class_name is None:
        raise DatusException(
            ErrorCode.COMMON_FIELD_INVALID,
            message=f"Unsupported model type: {model_cfg.type}",
        )
    module = __import__(f"datus.models.{model_cfg.type}_model", fromlist=[model_class_name])
    model_class = getattr(module, model_class_name)
    client = model_class(model_config=model_cfg)
    client.generate("Hello")


def _probe_datasource_sync(payload: Dict[str, Any]) -> None:
    """Build a one-shot connector from a raw dict and run a SELECT 1 probe."""
    from datus.tools.db_tools.db_manager import DBManager

    kwargs = dict(payload)
    kwargs.setdefault("name", "_probe_")
    db_config = DbConfig.filter_kwargs(DbConfig, kwargs)

    manager = DBManager({"_probe_": {"_probe_": db_config}})
    try:
        conn = manager.get_conn("_probe_")
        conn.test_connection()
    finally:
        manager.close()


def _validate_keys(entries: Dict[str, Any], kind: str) -> None:
    """Ensure every key matches the naming policy used by AgentConfig."""
    for name in entries.keys():
        if not _SAFE_NAME_RE.match(name):
            raise DatusException(
                ErrorCode.COMMON_FIELD_INVALID,
                message=(
                    f"Invalid {kind} name '{name}'. Only alphanumeric characters, underscores, and hyphens are allowed."
                ),
            )


async def _evict_current_project(project_id: str) -> None:
    """Drop the cached DatusService so the next request reloads from YAML."""
    cache = deps._service_cache
    if cache is None:
        return
    try:
        await cache.evict(project_id)
    except Exception:
        logger.exception(f"Failed to evict service cache for project {project_id}")


def _save_config_updates(
    cm: Any,
    data: Dict[str, Any],
    project_root: str,
    previous_override: Optional[ProjectOverride],
    updated_override: Optional[ProjectOverride],
) -> None:
    previous_data = cm.data
    cm.data = data
    try:
        if updated_override is not None:
            save_project_override(updated_override, cwd=project_root)
        cm.save()
    except Exception:
        cm.data = previous_data
        if updated_override is not None and previous_override is not None:
            try:
                save_project_override(previous_override, cwd=project_root)
            except Exception:
                logger.exception(f"Failed to roll back project override at {project_root}")
        raise


@router.get(
    "/config/agent",
    response_model=Result[dict],
    summary="Get Agent Configuration",
    description="Get the current project's agent configuration (models, datasource, agentic_nodes)",
)
async def get_agent_config_endpoint(
    svc: ServiceDep,
) -> Result[dict]:
    """Return the project's loaded AgentConfig summary."""
    config = svc.agent_config
    flat_datasources: dict = {}

    for db_name, db_config in config.datasource_configs.items():
        if db_config is None:
            continue
        flat_datasources[db_name] = db_config

    return Result(
        success=True,
        data={
            "target": config.target,
            "target_provider": getattr(config, "_target_provider", None),
            "target_model": getattr(config, "_target_model", None),
            "models": config.models,
            "current_datasource": config.current_datasource,
            "datasources": flat_datasources,
            "home": config.home,
        },
    )


def _namespace(db_config: Any, attr: str) -> Optional[str]:
    # Strings only: on a pydantic-shaped config, ``schema`` is BaseModel's method.
    value = getattr(db_config, attr, None)
    return value if isinstance(value, str) and value else None


@router.get(
    "/config/datasources",
    response_model=Result[DatasourceListData],
    summary="List Datasources",
    description="Names and types of the project's configured datasources. Carries no credentials.",
)
async def list_datasources_endpoint(
    svc: ServiceDep,
) -> Result[DatasourceListData]:
    """The datasource roster, for clients that need to name them.

    Separate from ``/config/agent`` on purpose. That endpoint returns the raw
    ``DbConfig`` of every datasource — decrypted password, credential-bearing
    ``uri``, Snowflake key passphrase — plus the models block with its API keys.
    Anything a catalog UI does on project entry has to be reachable without
    that, or every member who opens a project is handed the warehouse
    credentials.
    """
    config = svc.agent_config
    current = config.current_datasource or ""

    summaries = [
        DatasourceSummary(
            name=name,
            type=(db_config.type.value if hasattr(db_config.type, "value") else str(db_config.type)),
            is_current=(name == current),
            # Namespaces, not credentials: a client needs them to open a session on
            # the datasource's configured schema instead of the first one it lists.
            catalog=_namespace(db_config, "catalog"),
            database=_namespace(db_config, "database"),
            db_schema=_namespace(db_config, "schema"),
        )
        for name, db_config in config.datasource_configs.items()
        if db_config is not None
    ]

    return Result(
        success=True,
        data=DatasourceListData(datasources=summaries, current_datasource=current),
    )


@router.put(
    "/config/datasources",
    response_model=Result[dict],
    summary="Update Datasources",
    description="Replace the datasources (services.datasources) block in agent.yml.",
)
async def update_datasources_endpoint(
    body: UpdateDatasourcesRequest,
    svc: ServiceDep,  # noqa: ARG001  # populates request.state.app_context; must resolve before AppContextDep
    ctx: AppContextDep,
) -> Result[dict]:
    """Full-replace `services.datasources` with the provided datasources."""
    _validate_keys(body.datasources, kind="datasource")

    cm = configuration_manager()
    data = deepcopy(cm.data)
    services = data.setdefault("services", {})
    services["datasources"] = dict(body.datasources)

    override = load_project_override(cwd=svc.agent_config.project_root)
    updated_override = None
    if override is not None and override.default_datasource is not None:
        if override.default_datasource not in body.datasources:
            defaults = [name for name, config in body.datasources.items() if config.get("default") is True]
            updated_override = deepcopy(override)
            updated_override.default_datasource = defaults[0] if len(defaults) == 1 else None

    _save_config_updates(cm, data, svc.agent_config.project_root, override, updated_override)
    await _evict_current_project(ctx.project_id or "default")

    return Result(success=True, data={"updated": True})


@router.put(
    "/config/models",
    response_model=Result[dict],
    summary="Update Models and Target",
    description="Replace the models block and/or update the default target in agent.yml.",
)
async def update_models_endpoint(
    body: UpdateModelsRequest,
    svc: ServiceDep,  # noqa: ARG001
    ctx: AppContextDep,
) -> Result[dict]:
    """Optional full-replace `models`, optional update `target`. One must be set."""
    target_provided = "target" in body.model_fields_set
    if body.models is None and not target_provided:
        raise DatusException(
            ErrorCode.COMMON_FIELD_INVALID,
            message="At least one of 'models' or 'target' must be provided.",
        )

    if body.models is not None:
        _validate_keys(body.models, kind="model")

    cm = configuration_manager()

    if target_provided and body.target is not None:
        effective_models = body.models if body.models is not None else cm.data.get("models") or {}
        if body.target not in effective_models:
            raise DatusException(
                ErrorCode.COMMON_FIELD_INVALID,
                message=f"target '{body.target}' does not exist in models.",
            )

    data = deepcopy(cm.data)
    if body.models is not None:
        data["models"] = dict(body.models)
        current_target = data.get("target")
        if not target_provided and current_target and current_target not in body.models:
            replacement = next(iter(body.models), None)
            if replacement is None:
                data.pop("target", None)
            else:
                data["target"] = replacement
    if target_provided:
        if body.target is None:
            data.pop("target", None)
        else:
            data["target"] = body.target

    override = load_project_override(cwd=svc.agent_config.project_root)
    updated_override = None
    if override is not None and override.target is not None:
        if target_provided:
            updated_override = deepcopy(override)
            updated_override.target = ProjectTarget(custom=body.target) if body.target is not None else None
        elif body.models is not None:
            current_target = override.target if isinstance(override.target, str) else override.target.custom
            if current_target is not None and current_target not in body.models:
                replacement = next(iter(body.models), None)
                updated_override = deepcopy(override)
                updated_override.target = ProjectTarget(custom=replacement) if replacement is not None else None

    _save_config_updates(cm, data, svc.agent_config.project_root, override, updated_override)
    await _evict_current_project(ctx.project_id or "default")

    return Result(success=True, data={"updated": True})


@router.post(
    "/config/models/test",
    response_model=Result[dict],
    summary="Test Model Connectivity",
    description="Send a tiny probe to verify an LLM model config is reachable.",
)
async def probe_model_connectivity_endpoint(
    body: ProbeModelRequest,
    svc: ServiceDep,  # noqa: ARG001
) -> Result[dict]:
    """Return `{ok: True}` if the probe succeeds, else `{ok: False, message: ...}`."""
    payload = body.model_dump()
    try:
        await asyncio.to_thread(_probe_llm_sync, payload)
        return Result(success=True, data={"ok": True})
    except Exception as e:
        logger.info(f"Model connectivity probe failed: {e}")
        return Result(success=True, data={"ok": False, "message": str(e)})


@router.post(
    "/config/datasources/test",
    response_model=Result[dict],
    summary="Test Datasource Connectivity",
    description="Run SELECT 1 against a datasource config to verify reachability and credentials.",
)
async def probe_datasource_connectivity_endpoint(
    body: ProbeDatasourceRequest,
    svc: ServiceDep,  # noqa: ARG001
) -> Result[dict]:
    """Return `{ok: True}` if the probe succeeds, else `{ok: False, message: ...}`."""
    payload = body.model_dump()
    try:
        await asyncio.to_thread(_probe_datasource_sync, payload)
        return Result(success=True, data={"ok": True})
    except Exception as e:
        logger.info(f"Datasource connectivity probe failed: {e}")
        return Result(success=True, data={"ok": False, "message": str(e)})
